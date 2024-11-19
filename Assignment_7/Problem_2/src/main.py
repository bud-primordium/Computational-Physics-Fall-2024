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

    def convergence_study(self, n_points_list=None):
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
            # 默认测试点数列表：从100到10000，按2的幂次递增
            n_points_list = [100 * 2**i for i in range(7)]

        results = self.convergence_analyzer.analyze_grid_convergence(
            self, n_points_list
        )

        # 可视化结果
        self.visualizer.plot_convergence_study(results)

        return results

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
                E_min = -1.5 / (2 * self.config.n**2)
                E_max = -0.5 / (2 * self.config.n**2)
                E, _ = self.solver.shooting_solve(
                    E_min, E_max, self.config.n - self.config.l - 1
                )
                return E
            else:
                energies, _ = self.solver.solve(1)
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

    def solve(self) -> Dict:
        """求解量子态

        Returns
        -------
        Dict
            计算结果字典
        """
        try:
            # 能量范围估计
            E_min = -1.5 / (2 * self.config.n**2)
            E_max = -0.5 / (2 * self.config.n**2)

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
                energies, states = self.solver.solve(self.config.n_states)
                return {"energies": energies, "states": states}

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
        print("\n" + "=" * 60)
        print("进行网格收敛性分析")
        print("=" * 60)

        # 使用氢原子1s态作为测试案例
        config = SolverConfig(V_type="hydrogen", n=1, l=0, method="shooting")
        solver = RadialSchrodingerSolver(config)

        try:
            # 进行收敛性分析
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
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="径向薛定谔方程求解器")
    parser.add_argument(
        "--V-type", choices=["hydrogen", "lithium"], default="hydrogen", help="势能类型"
    )
    parser.add_argument("--n", type=int, default=1, help="主量子数")
    parser.add_argument("--l", type=int, default=0, help="角量子数")
    parser.add_argument(
        "--method", choices=["shooting", "fd"], default="fd", help="求解方法"
    )
    parser.add_argument("--example", action="store_true", help="运行示例计算")
    parser.add_argument("--convergence", action="store_true", help="进行网格收敛性分析")

    args = parser.parse_args()

    try:
        if args.convergence:
            # 创建配置
            config = SolverConfig(
                V_type=args.V_type, n=args.n, l=args.l, method=args.method
            )

            # 初始化求解器并进行收敛性分析
            solver = RadialSchrodingerSolver(config)
            results = solver.convergence_study()

            # 打印结果
            print("\n网格收敛性分析结果:")
            print(f"{'网格点数':>10} {'能量':>15} {'相对误差(%)':>15}")
            print("-" * 45)
            for n, E, err in zip(
                results["n_points"], results["energies"], results["errors"]
            ):
                print(f"{n:10d} {E:15.8f} {err:15.8f}")

        elif args.example:
            run_example()
        else:
            # 使用命令行参数创建配置
            config = SolverConfig(
                V_type=args.V_type, n=args.n, l=args.l, method=args.method
            )

            # 求解
            solver = RadialSchrodingerSolver(config)
            results = solver.solve()

            # 输出结果
            if "energy" in results:  # shooting方法
                E = results["energy"]
                analysis = results["analysis"]
                print(f"\n能量本征值: {E:.6f} Hartree")
                if analysis["theoretical_E"] is not None:
                    print(f"理论值: {analysis['theoretical_E']:.6f} Hartree")
                    print(f"相对误差: {analysis['relative_error']:.6f}%")
            elif "energies" in results:  # 有限差分法
                print("\n计算得到的能量本征值:")
                for i, E in enumerate(results["energies"]):
                    print(f"E_{i}: {E:.6f} Hartree")

                # 获取理论值进行对比
                theoretical = get_theoretical_values()[args.V_type]
                for i, E in enumerate(results["energies"]):
                    if (args.n, args.l) in theoretical:
                        theory = theoretical[(args.n, args.l)]
                        error = abs((E - theory) / theory) * 100
                        print(f"理论值: {theory:.6f} Hartree")
                        print(f"相对误差: {error:.6f}%")
                    break  # 只比较第一个能量值

    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")
        raise


if __name__ == "__main__":
    main()
