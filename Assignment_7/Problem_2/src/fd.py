"""径向薛定谔方程的有限差分求解"""

import numpy as np
from scipy.sparse import diags, linalg
from typing import Tuple
import logging

from base import RadialGrid, PotentialFunction

logger = logging.getLogger(__name__)


class FiniteDifferenceSolver:
    """有限差分法求解器

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

    def construct_hamiltonian(self):
        """构建哈密顿量矩阵

        Returns
        -------
        scipy.sparse.spmatrix
            哈密顿量稀疏矩阵

        Notes
        -----
        在非均匀网格上构建哈密顿量,包含:
        - 动能项(二阶导数)
        - 势能项
        - 向心势项
        """
        N = len(self.grid.r)
        delta = self.grid.config.delta

        # 非均匀网格上的二阶导数项
        d2_coef = np.exp(-2 * delta * self.grid.j) / (delta**2)
        d1_coef = -np.exp(-delta * self.grid.j) / (2 * delta)

        # 势能和角动量项
        diag = (
            -2 * d2_coef
            + self.V(self.grid.r)
            + self.l * (self.l + 1) / (2 * self.grid.r**2)
        )

        # 非对角项
        upper_diag = d2_coef[:-1] + d1_coef[:-1]
        lower_diag = d2_coef[:-1] - d1_coef[:-1]

        # 构建稀疏矩阵
        H = diags([lower_diag, diag, upper_diag], [-1, 0, 1], format="csr")

        # 边界条件
        H[0, :] = 0
        H[0, 0] = 1.0
        H[-1, :] = 0
        H[-1, -1] = 1.0

        return H

    def solve(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """求解本征值问题

        Parameters
        ----------
        n_states : int
            需要求解的本征态数量

        Returns
        -------
        ndarray
            本征能量数组
        ndarray
            本征态波函数数组,每列为一个本征态

        Notes
        -----
        使用稀疏矩阵求解器计算最低的n_states个本征态
        """
        # 构建哈密顿量
        H = self.construct_hamiltonian()

        # 求解本征值问题
        energies, states = linalg.eigsh(
            H, k=n_states, which="SA", return_eigenvectors=True
        )

        # 按能量排序
        idx = np.argsort(energies)
        energies = energies[idx]
        states = states[:, idx]

        # 归一化波函数
        for i in range(states.shape[1]):
            norm = np.sqrt(np.trapz(states[:, i] ** 2, self.grid.r))
            states[:, i] /= norm

            # 调整波函数符号,使第一个非零值为正
            nonzero_idx = np.nonzero(np.abs(states[:, i]) > 1e-10)[0]
            if len(nonzero_idx) > 0 and states[nonzero_idx[0], i] < 0:
                states[:, i] *= -1

        return energies, states

    @staticmethod
    def check_convergence(
        energies: np.ndarray, ref_energies: np.ndarray, rtol: float = 1e-6
    ) -> bool:
        """检查能量收敛性

        Parameters
        ----------
        energies : ndarray
            当前计算的能量
        ref_energies : ndarray
            参考能量值
        rtol : float, optional
            相对误差容限, 默认1e-6

        Returns
        -------
        bool
            是否收敛
        """
        if len(energies) != len(ref_energies):
            return False

        rel_errors = np.abs((energies - ref_energies) / ref_energies)
        return np.all(rel_errors < rtol)
