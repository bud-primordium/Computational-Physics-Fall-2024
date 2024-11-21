"""径向薛定谔方程求解主程序

整合各功能模块，提供完整的计算流程和用户接口。

功能特点:
- 命令行参数配置和运行控制
- 多种原子、量子态和求解方法的示例
- 自动化的收敛性分析流程
- 结构化的结果输出和可视化

Modules:
    utils: 工具函数和配置类
    solver: 数值求解方法
    analysis: 结果分析和处理 
    visualization: 结果可视化
"""

import logging
import argparse
import os
import sys
from typing import Dict


# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radial_schrodinger.utils import (
    SolverConfig,
    RadialGrid,
    PotentialFunction,
    WavefunctionTools,
    get_theoretical_values,
    get_energy_bounds,
)
from radial_schrodinger.solver import ShootingSolver, FiniteDifferenceSolver
from radial_schrodinger.analysis import (
    WavefunctionProcessor,
    EnergyAnalyzer,
    ConvergenceAnalyzer,
)
from radial_schrodinger.visualization import ResultVisualizer


logger = logging.getLogger(__name__)


class RadialSchrodingerSolver:
    """径向薛定谔方程主求解器

    整合配置、网格、求解器、分析器和可视化器，
    提供完整的求解流程控制。

    Attributes
    ----------
    config : SolverConfig
        计算配置
    grid : RadialGrid
        计算网格
    V : callable
        势能函数
    wave_processor : WavefunctionProcessor
        波函数处理器
    energy_analyzer : EnergyAnalyzer
        能量分析器
    visualizer : ResultVisualizer
        结果可视化器
    convergence_analyzer : ConvergenceAnalyzer
        收敛性分析器
    solver : Union[ShootingSolver, FiniteDifferenceSolver]
        具体求解器实例
    """

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
                n_points_list = [50, 100, 150, 200, 300]
            else:
                n_points_list = [100 * 2**i for i in range(7)]  # 到6400

        results = self.convergence_analyzer.analyze_grid_convergence(
            self, n_points_list
        )

        # 可视化结果
        self.visualizer.plot_convergence_study(
            results,
            self.config.V_type,
            self.config.n,
            self.config.l,
            self.config.method,
        )

    def solve(self) -> Dict:
        """求解指定的量子态

        根据配置选择求解方法，完成求解并进行后处理：
        1. 确定能量搜索范围
        2. 求解本征值和本征函数
        3. 波函数归一化和处理
        4. 与理论值比较
        5. 结果可视化

        Returns
        -------
        Dict
            计算结果字典，包含:
            - energy: 本征能量
            - wavefunction: 波函数数据
            - analysis: 与理论值的比较
            - all_energies: (仅FD方法)所有本征值
            - all_states: (仅FD方法)所有本征态

        Raises
        ------
        Exception
            求解过程中的错误
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
                    method="shooting",
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
                    method="fd",
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
                n_points_list = [50, 100, 150, 200, 300]
            else:
                n_points_list = [100, 200, 400, 800, 1600, 3200]

            results = solver.convergence_study(n_points_list)

            # 只在results不为None且包含必要数据时打印结果
            if (
                results
                and "n_points" in results
                and "energies" in results
                and "errors" in results
            ):
                print("\n网格收敛性分析结果:")
                print(f"{'网格点数':>10} {'能量':>15} {'相对误差(%)':>15}")
                print("-" * 45)
                for n, E, err in zip(
                    results["n_points"], results["energies"], results["errors"]
                ):
                    print(f"{n:10d} {E:15.8f} {err:15.8f}")

        except KeyError as e:
            logger.debug(f"结果数据结构不完整: {str(e)}")
        except Exception as e:
            logger.error(f"收敛性分析出现错误: {str(e)}")
            continue


def main():
    """主程序入口

    处理命令行参数并执行相应的计算路径:
    1. 示例计算模式
    2. 收敛性分析模式
    3. 单次计算模式

    支持的参数:
    --V-type: 势能类型(hydrogen/lithium)
    --n: 主量子数
    --l: 角量子数
    --method: 求解方法(shooting/fd)
    --j-max: 网格点数
    --example: 运行示例
    --convergence: 进行收敛性分析
    """
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

    try:
        # 3. 根据命令行参数选择执行路径
        if args.example:
            # 路径1：运行示例计算
            run_example()

        elif args.convergence:
            # 路径2：进行收敛性分析
            # 创建配置
            config_dict = {
                "V_type": args.V_type,
                "n": args.n,
                "l": args.l,
                "method": args.method,
            }
            if args.j_max is not None:
                config_dict["j_max"] = args.j_max

            config = SolverConfig(**config_dict)

            # 打印分析信息
            print("\n" + "=" * 60)
            print("进行网格收敛性分析")
            print("=" * 60)
            print(config.config_summary)

            # 执行收敛性分析
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

        else:
            # 路径3：执行单次计算
            config_dict = {
                "V_type": args.V_type,
                "n": args.n,
                "l": args.l,
                "method": args.method,
            }
            if args.j_max is not None:
                config_dict["j_max"] = args.j_max

            config = SolverConfig(**config_dict)

            # 打印计算信息
            print("\n" + "=" * 60)
            print("进行单次求解计算")
            print("=" * 60)
            print(config.config_summary)

            # 执行计算
            solver = RadialSchrodingerSolver(config)
            results = solver.solve()

            # 打印结果
            if "energy" in results:
                E = results["energy"]
                analysis = results["analysis"]
                print(f"\n能量本征值: {E:.6f} Hartree")
                if analysis["theoretical_E"] is not None:
                    print(f"理论值: {analysis['theoretical_E']:.6f} Hartree")
                    print(f"相对误差: {analysis['relative_error']:.6f}%")

    except Exception as e:
        logger.error(f"程序运行失败: {str(e)}")


if __name__ == "__main__":
    main()
