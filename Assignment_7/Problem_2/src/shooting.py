"""径向薛定谔方程的打靶法求解"""

import numpy as np
from scipy.optimize import root_scalar
from typing import Tuple
import logging

from base import SolverConfig, RadialGrid, PotentialFunction

logger = logging.getLogger(__name__)


class ShootingSolver:
    """打靶法求解器

    从外向内积分求解径向薛定谔方程。使用RK4方法求解变换后的方程:
    d²v/dj² - (δ²/4)v = r_p²δ²e^(2jδ)[V(r) - E - l(l+1)/(2r²)]v

    Parameters
    ----------
    grid : RadialGrid
        径向网格对象
    V : callable
        势能函数
    l : int
        角量子数
    """

    def __init__(self, grid: RadialGrid, V: callable, l: int):
        self.grid = grid
        self.V = V
        self.l = l
        self.delta = grid.config.delta

    def integrate_inward(self, E: float) -> np.ndarray:
        """从外向内积分波函数

        Parameters
        ----------
        E : float
            能量尝试值

        Returns
        -------
        ndarray
            径向波函数u(r)

        Notes
        -----
        使用RK4方法求解变换后的方程。在靠近原子核处使用自适应步长。
        """
        # 初始化波函数
        v = np.zeros(self.grid.config.j_max + 1, dtype=np.float64)
        dvdj = np.zeros(self.grid.config.j_max + 1, dtype=np.float64)

        # 边界条件
        v[-1] = 0.0
        v[-2] = 1e-12  # 使用小的初始值以提高数值稳定性

        def get_effective_potential(r: float) -> float:
            """计算包含向心势的有效势能"""
            if r < 1e-10:
                return -E + 1e10  # 原点处使用大的排斥势
            return self.V(r) - E - self.l * (self.l + 1) / (2 * r * r)

        def d2vdj2(j: float, v_val: float) -> float:
            """计算波函数的二阶导数"""
            r = self.grid.r_p * (np.exp(self.delta * j) - 1)
            exp_term = np.exp(2 * self.delta * j)
            # 限制指数项增长
            exp_term = min(exp_term, 1e30)

            V_eff = get_effective_potential(r)
            return (
                (self.delta**2 / 4) * v_val
                - self.grid.r_p**2 * self.delta**2 * exp_term * V_eff * v_val
            )

        # 初始步长
        h = -self.delta  # 负步长，向内积分
        base_step = h

        # RK4向内积分
        for j in range(self.grid.config.j_max - 1, -1, -1):
            r = self.grid.r[j]

            # 自适应步长
            if r < 1.0:
                h = base_step * min(1.0, r)

            try:
                # RK4步骤
                k1v = h * dvdj[j + 1]
                k1dv = h * d2vdj2(j + 1, v[j + 1])

                k2v = h * (dvdj[j + 1] + 0.5 * k1dv)
                k2dv = h * d2vdj2(j + 0.5, v[j + 1] + 0.5 * k1v)

                k3v = h * (dvdj[j + 1] + 0.5 * k2dv)
                k3dv = h * d2vdj2(j + 0.5, v[j + 1] + 0.5 * k2v)

                k4v = h * (dvdj[j + 1] + k3dv)
                k4dv = h * d2vdj2(j, v[j + 1] + k3v)

                # 更新波函数和导数
                v_new = v[j + 1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
                dvdj_new = dvdj[j + 1] + (k1dv + 2 * k2dv + 2 * k3dv + k4dv) / 6

                # 数值稳定性检查
                if np.isnan(v_new) or np.abs(v_new) > 1e15:
                    logger.warning(f"数值不稳定 at j={j}, r={r:.6e}")
                    v_new = v[j + 1] * 0.9
                    dvdj_new = dvdj[j + 1] * 0.9

                v[j] = v_new
                dvdj[j] = dvdj_new

            except Exception as e:
                logger.error(f"积分过程出错 at j={j}, r={r:.6e}: {str(e)}")
                raise RuntimeError(f"积分不稳定 at j={j}, r={r}")

        # 变换回u(r)
        exp_factor = np.exp(self.delta * self.grid.j / 2)
        exp_factor = np.minimum(exp_factor, 1e30)
        u = v * exp_factor

        # 波函数归一化
        if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e20):
            logger.warning("波函数包含不稳定值，进行归一化处理")
            u = np.nan_to_num(u, nan=0.0, posinf=1e20, neginf=-1e20)
            norm = np.sqrt(np.sum(u * u) * self.delta)
            if norm > 0:
                u /= norm

        return u

    def shooting_solve(
        self, E_min: float, E_max: float, target_nodes: int
    ) -> Tuple[float, np.ndarray]:
        """打靶法求解本征值和本征函数

        Parameters
        ----------
        E_min : float
            能量搜索下限
        E_max : float
            能量搜索上限
        target_nodes : int
            目标节点数

        Returns
        -------
        float
            能量本征值
        ndarray
            对应的波函数

        Raises
        ------
        RuntimeError
            求解未收敛或初始条件不当
        """

        def objective(E: float) -> float:
            """目标函数：计算r=0处的值并考虑节点数"""
            u = self.integrate_inward(E)
            nodes = self._count_nodes(u)

            if nodes != target_nodes:
                return 1e3 * (nodes - target_nodes)  # 节点数不对时施加惩罚
            return u[0]

        try:
            # 使用brentq方法求解
            result = root_scalar(
                objective,
                method="brentq",
                bracket=[E_min, E_max],
                rtol=1e-6,
                maxiter=1000,
            )

            if not result.converged:
                # 尝试不同的初始能量范围
                E_min_new = E_min * 1.5
                E_max_new = E_max * 0.5
                result = root_scalar(
                    objective,
                    method="brentq",
                    bracket=[E_min_new, E_max_new],
                    rtol=1e-6,
                    maxiter=1000,
                )

        except ValueError as e:
            logger.error(f"能量求解出错: {str(e)}")
            raise RuntimeError("能量求解失败，请检查能量范围和初始条件")

        if not result.converged:
            raise RuntimeError("能量求解未收敛!")

        # 计算最终波函数
        E = result.root
        u = self.integrate_inward(E)

        return E, u

    @staticmethod
    def _count_nodes(u: np.ndarray, eps: float = 1e-10) -> int:
        """计算波函数节点数

        Parameters
        ----------
        u : ndarray
            波函数数组
        eps : float, optional
            判定零点的阈值, 默认1e-10

        Returns
        -------
        int
            节点数
        """
        # 忽略小波动
        u_filtered = np.where(abs(u) < eps, 0, u)
        # 寻找符号变化点
        return len(np.where(np.diff(np.signbit(u_filtered)))[0])
