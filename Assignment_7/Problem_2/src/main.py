"""径向薛定谔方程求解器主程序"""

import numpy as np
import logging
from typing import Optional, List

from base import SolverConfig, RadialGrid, PotentialFunction
from shooting import ShootingSolver
from fd import FiniteDifferenceSolver
from analysis import WavefunctionAnalyzer, ResultVisualizer

logger = logging.getLogger(__name__)


class RadialSchrodinger:
    """径向薛定谔方程求解器

    Parameters
    ----------
    config : SolverConfig
        求解器配置
    """

    def __init__(self, config: SolverConfig):
        config.validate()
        self.config = config

        # 初始化网格
        self.grid = RadialGrid(config)

        # 获取势能函数
        self.V = PotentialFunction.get_potential(config.V_type)

        # 初始化分析工具
        self.analyzer = WavefunctionAnalyzer(self.grid, config.l)
        self.visualizer = ResultVisualizer(self.grid)

        # 选择求解器
        if config.method == "shooting":
            self.solver = ShootingSolver(self.grid, self.V, config.l)
        else:
            self.solver = FiniteDifferenceSolver(self.grid, self.V, config.l)

    def solve(self) -> dict:
        """求解方程并分析结果

        Returns
        -------
        dict
            计算结果字典，包含波函数、能量和分析数据
        """
        try:
            if self.config.method == "shooting":
                # 打靶法求解单个态
                target_nodes = self.config.n - self.config.l - 1
                E_min = -10.0 / self.config.n**2  # 估计能量范围
                E_max = -0.01 / self.config.n**2

                E, u = self.solver.shooting_solve(E_min, E_max, target_nodes)
                u_norm, R = self.analyzer.normalize(u)

                # 获取解析解（如果有）
                R_analytic = None
                if self.config.V_type == "hydrogen":
                    R_analytic = self.analyzer.get_analytic_hydrogen(self.config.n)

                # 分析结果
                analysis = self.analyzer.analyze_state(u_norm, R, E)

                # 绘图
                self.visualizer.plot_wavefunction(
                    u_norm, R, E, self.config.n, self.config.l, R_analytic
                )

                return {
                    "energy": E,
                    "wavefunction": {"u": u_norm, "R": R, "R_analytic": R_analytic},
                    "analysis": analysis,
                }

            else:
                # 有限差分法求解多个态
                energies, states = self.solver.solve(self.config.n_states)
                results = []

                for i, (E, state) in enumerate(zip(energies, states.T)):
                    u_norm, R = self.analyzer.normalize(state)

                    # 获取解析解
                    n_state = i + self.config.l + 1
                    R_analytic = None
                    if self.config.V_type == "hydrogen":
                        R_analytic = self.analyzer.get_analytic_hydrogen(n_state)

                    # 分析结果
                    analysis = self.analyzer.analyze_state(u_norm, R, E)

                    # 绘图
                    self.visualizer.plot_wavefunction(
                        u_norm, R, E, n_state, self.config.l, R_analytic
                    )

                    results.append(
                        {
                            "n": n_state,
                            "energy": E,
                            "wavefunction": {
                                "u": u_norm,
                                "R": R,
                                "R_analytic": R_analytic,
                            },
                            "analysis": analysis,
                        }
                    )

                return {"states": results, "energies": energies}

        except Exception as e:
            logger.error(f"求解过程出错: {str(e)}")
            raise


def run_examples():
    """运行示例计算"""

    # 示例1: 氢原子基态（打靶法）
    print("\n求解氢原子基态（1s）...")
    config_h_1s = SolverConfig(V_type="hydrogen", n=1, l=0, method="shooting")
    solver = RadialSchrodinger(config_h_1s)
    result = solver.solve()
    print(f"基态能量: {result['energy']:.6f} Hartree")
    print(f"节点数: {result['analysis']['nodes']}")

    # 示例2: 氢原子2p态（打靶法）
    print("\n求解氢原子2p态...")
    config_h_2p = SolverConfig(V_type="hydrogen", n=2, l=1, method="shooting")
    solver = RadialSchrodinger(config_h_2p)
    result = solver.solve()
    print(f"2p能量: {result['energy']:.6f} Hartree")

    # 示例3: 锂原子基态（有限差分法）
    print("\n求解锂原子前三个s态...")
    config_li = SolverConfig(V_type="lithium", n=1, l=0, n_states=3, method="fd")
    solver = RadialSchrodinger(config_li)
    result = solver.solve()
    for i, state in enumerate(result["states"]):
        print(f"{i+1}s能量: {state['energy']:.6f} Hartree")


if __name__ == "__main__":
    try:
        run_examples()
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        raise
