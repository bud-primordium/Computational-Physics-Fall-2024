"""径向薛定谔方程求解主程序

提供命令行接口和示例运行功能。
整合求解、分析和可视化模块，实现完整的求解流程。
"""

import logging
import argparse
import os
import sys
from typing import Dict


# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    SolverConfig,
    RadialGrid,
    PotentialFunction,
    WavefunctionTools,
    get_theoretical_values,
    get_energy_bounds,
)
from src.solver import ShootingSolver, FiniteDifferenceSolver
from src.analysis import WavefunctionProcessor, EnergyAnalyzer, ConvergenceAnalyzer
from src.visualization import ResultVisualizer


logger = logging.getLogger(__name__)


class RadialSchrodingerSolver:
    """径向薛定谔方程主求解器"""

    def __init__(self, config: SolverConfig):
        """初始化求解器

        Parameters
        ----------
        config : SolverConfig
            求解器配置
        """
        self.config = config

        # 初始化组件
        self.grid = RadialGrid(config)
        self.V = PotentialFunction.get_potential(config.V_type)
        self.wave_processor = WavefunctionProcessor(self.grid.r, config.l, config.delta)
        self.energy_analyzer = EnergyAnalyzer(get_theoretical_values())
        self.visualizer = ResultVisualizer(self.grid.r)
        self.convergence_analyzer = ConvergenceAnalyzer(self.energy_analyzer)

        # 选择求解器
        if config.method == "shooting":
            self.solver = ShootingSolver(self.grid, self.V, config.l)
        else:
            self.solver = FiniteDifferenceSolver(self.grid, self.V, config.l)

        logger.info(
            f"初始化{config.V_type}原子求解器: "
            f"n={config.n}, l={config.l}, 方法={config.method}"
        )

    def solve_with_points(self, n_points: int) -> float:
        """使用指定网格点数求解

        Parameters
        ----------
        n_points : int
            网格点数

        Returns
        -------
        float
            计算得到的能量
        """
        # 保存原始配置
        original_j_max = self.config.j_max

        try:
            # 更新网格点数
            self.config.j_max = n_points
            # 重新初始化网格
            self.grid = RadialGrid(self.config)

            # 重新初始化求解器
            if self.config.method == "shooting":
                self.solver = ShootingSolver(self.grid, self.V, self.config.l)
            else:
                self.solver = FiniteDifferenceSolver(self.grid, self.V, self.config.l)

            # 求解
            if self.config.method == "shooting":
                # 获取能量范围估计
                E_min, E_max = get_energy_bounds(self.config.V_type, self.config.n)
                E, _ = self.solver.shooting_solve(
                    E_min, E_max, self.config.n - self.config.l - 1
                )
                return E
            else:
                energies, _ = self.solver.fd_solve(1)
                return energies[0]

        finally:
            # 恢复原始配置
            self.config.j_max = original_j_max
            self.grid = RadialGrid(self.config)
            # 重新初始化求解器
            if self.config.method == "shooting":
                self.solver = ShootingSolver(self.grid, self.V, self.config.l)
            else:
                self.solver = FiniteDifferenceSolver(self.grid, self.V, self.config.l)

    def convergence_study(self, n_points_list=None) -> Dict:
        """进行收敛性分析

        Parameters
        ----------
        n_points_list : list, optional
            要测试的网格点数列表

        Returns
        -------
        dict
            收敛性分析结果
        """
        if n_points_list is None:
            if self.config.method == "fd":
                n_points_list = [100, 200, 300, 400, 500, 600]
            else:
                n_points_list = [100 * 2**i for i in range(7)]  # 到6400

        results = self.convergence_analyzer.analyze_grid_convergence(
            self, n_points_list
        )

        # 可视化结果
        self.visualizer.plot_convergence_study(results)

    def solve(self) -> Dict:
        """求解量子态

        Returns
        -------
        Dict
            计算结果字典
        """
        try:
            # 获取能量范围估计
            E_min, E_max = get_energy_bounds(self.config.V_type, self.config.n)

            if self.config.method == "shooting":
                # 打靶法求解
                E, u = self.solver.shooting_solve(
                    E_min, E_max, self.config.n - self.config.l - 1
                )
                # 处理波函数
                u_norm, R = self.wave_processor.normalize_wavefunction(u)

                # 分析结果
                analysis = self.energy_analyzer.compare_with_theory(
                    E, self.config.V_type, self.config.n, self.config.l
                )

                # 获取解析解(如果有)
                R_analytic = None
                if self.config.V_type == "hydrogen":
                    R_analytic = WavefunctionTools.get_analytic_hydrogen(
                        self.grid.r, self.config.n, self.config.l
                    )

                # 可视化
                self.visualizer.plot_wavefunction(
                    u_norm,
                    R,
                    E,
                    self.config.n,
                    self.config.l,
                    self.config.V_type,
                    R_analytic,
                )

                return {
                    "energy": E,
                    "wavefunction": {"u": u_norm, "R": R, "R_analytic": R_analytic},
                    "analysis": analysis,
                }

            else:
                # 有限差分法求解
                energies, states = self.solver.fd_solve(self.config.n_states)

                # 选择指定(n,l)对应的态
                state_idx = 0  # 默认取最低能态
                # 如果n和l指定的不是基态，需要找到对应的激发态
                if self.config.n > 1:
                    state_idx = self.config.n - self.config.l - 1

                E = energies[state_idx]
                u = states[:, state_idx]

                # 处理波函数
                u_norm, R = self.wave_processor.normalize_wavefunction(u)

                # 分析结果
                analysis = self.energy_analyzer.compare_with_theory(
                    E, self.config.V_type, self.config.n, self.config.l
                )

                # 获取解析解(如果有)
                R_analytic = None
                if self.config.V_type == "hydrogen":
                    R_analytic = WavefunctionTools.get_analytic_hydrogen(
                        self.grid.r, self.config.n, self.config.l
                    )

                # 可视化
                self.visualizer.plot_wavefunction(
                    u_norm,
                    R,
                    E,
                    self.config.n,
                    self.config.l,
                    self.config.V_type,
                    R_analytic,
                )

                # 返回与shooting方法相同格式的结果
                result = {
                    "energy": E,
                    "wavefunction": {"u": u_norm, "R": R, "R_analytic": R_analytic},
                    "analysis": analysis,
                }

                # 额外添加所有本征态的信息
                result.update({"all_energies": energies, "all_states": states})

                return result

        except Exception as e:
            logger.error(f"求解失败: {str(e)}")
            raise


def run_example():
    """运行示例计算"""
    # 配置示例
    configs = [
        # 氢原子基态
        SolverConfig(V_type="hydrogen", n=1, l=0, method="shooting"),
        # 氢原子2s态
        SolverConfig(V_type="hydrogen", n=2, l=0, method="shooting"),
        # 氢原子2p态
        SolverConfig(V_type="hydrogen", n=2, l=1, method="shooting"),
        # 锂原子基态
        SolverConfig(V_type="lithium", n=1, l=0, method="shooting"),
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"求解{config.V_type}原子: n={config.n}, l={config.l}")
        print(f"使用{config.method}方法")
        print("=" * 60)

        try:
            solver = RadialSchrodingerSolver(config)
            results = solver.solve()

            # 打印能量
            if "energy" in results:
                E = results["energy"]
                analysis = results["analysis"]
                print(f"\n能量本征值: {E:.6f} Hartree")
                if analysis["theoretical_E"] is not None:
                    print(f"理论值: {analysis['theoretical_E']:.6f} Hartree")
                    print(f"相对误差: {analysis['relative_error']:.6f}%")

        except Exception as e:
            logger.error(f"示例运行失败: {str(e)}")
            continue

    # 添加收敛性分析示例
    methods = ["shooting", "fd"]
    for method in methods:
        print("\n" + "=" * 60)
        print(f"进行氢原子1s态 {method}方法 网格收敛性分析")
        print("=" * 60)

        # 使用氢原子1s态作为测试案例
        config = SolverConfig(V_type="hydrogen", n=1, l=0, method=method)
        solver = RadialSchrodingerSolver(config)

        try:
            # 选择合适的网格点序列
            if method == "fd":
                n_points_list = [100, 200, 300, 400, 500, 600]
            else:
                n_points_list = [100, 200, 400, 800, 1600, 3200]

            results = solver.convergence_study(n_points_list)

            # 打印分析结果
            print("\n网格收敛性分析结果:")
            print(f"{'网格点数':>10} {'能量':>15} {'相对误差(%)':>15}")
            print("-" * 45)
            for n, E, err in zip(
                results["n_points"], results["energies"], results["errors"]
            ):
                print(f"{n:10d} {E:15.8f} {err:15.8f}")

        except Exception as e:
            logger.error(f"收敛性分析失败: {str(e)}")


def main():
    """主函数"""
    # 1. 设置日志系统
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 2. 命令行参数解析
    parser = argparse.ArgumentParser(description="径向薛定谔方程求解器")
    parser.add_argument(
        "--V-type", choices=["hydrogen", "lithium"], default="hydrogen", help="势能类型"
    )
    parser.add_argument("--n", type=int, default=1, help="主量子数")
    parser.add_argument("--l", type=int, default=0, help="角量子数")
    parser.add_argument(
        "--method", choices=["shooting", "fd"], default="shooting", help="求解方法"
    )
    parser.add_argument("--j-max", type=int, help="网格点数(可选)")
    parser.add_argument("--example", action="store_true", help="运行示例计算")
    parser.add_argument("--convergence", action="store_true", help="进行网格收敛性分析")

    args = parser.parse_args()


if __name__ == "__main__":
    main()
