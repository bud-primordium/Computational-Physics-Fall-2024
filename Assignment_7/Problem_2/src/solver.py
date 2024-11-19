"""径向薛定谔方程求解器的核心求解模块

实现了两种求解方法:
1. 打靶法(shooting): 从外向内积分,寻找满足边界条件的能量本征值
2. 有限差分法(finite difference): 构建矩阵直接求解本征值问题

Classes:
    ShootingSolver: 打靶法求解器
    FiniteDifferenceSolver: 有限差分法求解器
"""

import numpy as np
from scipy.sparse import linalg, lil_matrix
from scipy.optimize import root_scalar
from typing import Tuple
import logging

from .utils import RadialGrid, WavefunctionTools

logger = logging.getLogger(__name__)


class ShootingSolver:
    """打靶法求解器

    使用改进的RK4方法从外向内积分求解径向方程
    """

    def __init__(self, grid: RadialGrid, V: callable, l: int):
        """初始化求解器

        Parameters
        ----------
        grid : RadialGrid
            网格对象
        V : callable
            势能函数
        l : int
            角量子数
        """
        self.grid = grid
        self.V = V
        self.l = l
        self.delta = grid.config.delta

    def integrate_inward(self, E: float) -> np.ndarray:
        """从外向内积分

        Parameters
        ----------
        E : float
            能量

        Returns
        -------
        np.ndarray
            波函数u(r)
        """
        v = np.zeros(self.grid.config.j_max + 1)
        dvdj = np.zeros_like(v)

        # 边界条件
        v[-1] = 0.0
        v[-2] = 1e-12

        def V_eff(j):
            # 计算对应的 r 值
            r = self.grid.r_p * (np.exp(self.delta * j) - 1) + self.grid.config.r_min
            return self.V(r) + self.l * (self.l + 1) / (2 * r * r)

        # RK4积分
        h = -self.delta
        for j in range(self.grid.config.j_max - 1, -1, -1):
            r = self.grid.r[j]

            # 自适应步长
            min_step = 1e-5
            h = -max(self.delta * min(1.0, r), min_step)

            # RK4 steps
            # k1
            k1v = h * dvdj[j + 1]
            k1dv = h * (
                -self.delta**2 / 4 * v[j + 1]
                + 2
                * self.grid.r_p**2
                * self.delta**2
                * np.exp(2 * self.delta * j)
                * (V_eff(j) - E)
                * v[j + 1]
            )

            # k2
            k2v = h * (dvdj[j + 1] + 0.5 * k1dv)
            k2dv = h * (
                -self.delta**2 / 4 * (v[j + 1] + 0.5 * k1v)
                + 2
                * self.grid.r_p**2
                * self.delta**2
                * np.exp(2 * self.delta * (j + 0.5))
                * (V_eff(j + 0.5) - E)
                * (v[j + 1] + 0.5 * k1v)
            )

            # k3
            k3v = h * (dvdj[j + 1] + 0.5 * k2dv)
            k3dv = h * (
                -self.delta**2 / 4 * (v[j + 1] + 0.5 * k2v)
                + 2
                * self.grid.r_p**2
                * self.delta**2
                * np.exp(2 * self.delta * (j + 0.5))
                * (V_eff(j + 0.5) - E)
                * (v[j + 1] + 0.5 * k2v)
            )

            # k4
            k4v = h * (dvdj[j + 1] + k3dv)
            k4dv = h * (
                -self.delta**2 / 4 * (v[j + 1] + k3v)
                + 2
                * self.grid.r_p**2
                * self.delta**2
                * np.exp(2 * self.delta * j)
                * (V_eff(j) - E)
                * (v[j + 1] + k3v)
            )

            v[j] = v[j + 1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            dvdj[j] = dvdj[j + 1] + (k1dv + 2 * k2dv + 2 * k3dv + k4dv) / 6

        # 变换回u(r)
        u = v * np.exp(self.delta * self.grid.j / 2)
        return u

    def shooting_solve(
        self, E_min: float, E_max: float, target_nodes: int
    ) -> Tuple[float, np.ndarray]:
        """打靶法求解本征值和本征函数

        Parameters
        ----------
        E_min, E_max : float
            能量搜索范围
        target_nodes : int
            目标节点数

        Returns
        -------
        float
            能量本征值
        np.ndarray
            本征函数
        """

        def objective(E: float) -> float:
            u = self.integrate_inward(E)
            nodes = WavefunctionTools.count_nodes(u)
            if nodes != target_nodes:
                return 1e3 * (nodes - target_nodes)
            return u[0]

        try:
            result = root_scalar(
                objective, bracket=[E_min, E_max], method="brentq", rtol=1e-6
            )
            if result.converged:
                E = result.root
                u = self.integrate_inward(E)
                return E, u
            raise RuntimeError("能量求解未收敛")

        except Exception as e:
            logger.error(f"求解失败: {str(e)}")
            raise


class FiniteDifferenceSolver:
    """有限差分法求解器"""

    def __init__(self, grid: RadialGrid, V: callable, l: int):
        """初始化求解器

        Parameters
        ----------
        grid : RadialGrid
            网格对象
        V : callable
            势能函数
        l : int
            角量子数
        """
        self.grid = grid
        self.V = V
        self.l = l

    def construct_hamiltonian(self):
        """构建哈密顿量矩阵

        Returns
        -------
        scipy.sparse.spmatrix
            哈密顿量稀疏矩阵
        """
        N = len(self.grid.r)
        delta = self.grid.config.delta

        # 动能项
        d2_coef = np.exp(-2 * delta * self.grid.j) / (delta**2)
        d1_coef = -np.exp(-delta * self.grid.j) / (2 * delta)

        # 势能和角动量项
        diag = (
            -2 * d2_coef
            + self.V(self.grid.r)
            + self.l * (self.l + 1) / (2 * self.grid.r**2)
        )

        # 非对角项
        upper = d2_coef[:-1] + d1_coef[:-1]
        lower = d2_coef[:-1] - d1_coef[:-1]

        # 使用 LIL 格式构建稀疏矩阵
        H = lil_matrix((N, N))

        # 设置对角线元素
        H.setdiag(diag)

        # 设置上下对角线元素
        H.setdiag(lower, k=-1)
        H.setdiag(upper, k=1)

        # 边界条件
        H[0, :] = 0
        H[-1, :] = 0
        H[0, 0] = 1.0
        H[-1, -1] = 1.0

        # 转换为 CSR 格式
        H = H.tocsr()

        return H

    def solve(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """求解本征值问题

        Parameters
        ----------
        n_states : int
            需要求解的本征态数量

        Returns
        -------
        np.ndarray
            本征能量
        np.ndarray
            本征态波函数
        """
        H = self.construct_hamiltonian()

        try:
            # 求解本征值问题
            energies, states = linalg.eigsh(H, k=n_states, which="SA")

            # 排序和归一化
            idx = np.argsort(energies)
            energies = energies[idx]
            states = states[:, idx]

            for i in range(states.shape[1]):
                norm = np.sqrt(np.trapz(states[:, i] ** 2, self.grid.r))
                if norm > 0:
                    states[:, i] /= norm

            return energies, states

        except Exception as e:
            logger.error(f"本征值求解失败: {str(e)}")
            raise
