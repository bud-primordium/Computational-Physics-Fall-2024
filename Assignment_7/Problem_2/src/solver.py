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

from src.utils import RadialGrid, WavefunctionTools

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
        # 保存一些常用值
        self.r_p = grid.r_p
        self.j = grid.j

    def V_eff(self, j) -> float:
        """计算有效势

        Parameters
        ----------
        j : 不只局限于整数
            当前的网格索引，可以是半整数，便于rK4积分

        Returns
        -------
        float
            有效势
        """
        # 计算对应的 r 值
        r = self.r_p * (np.exp(self.delta * j) - 1) + self.grid.config.r_min
        # 对离心势也截断一个safe，考虑到比1/r更高次，截断在1e-8
        r_safe = np.maximum(r, 1e-8)  # 同时支持标量和数组
        return self.V(r_safe) + self.l * (self.l + 1) / (2 * r_safe * r_safe)

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
        v[-1] = 0
        dvdj[-1] = -1  # 从外向内积分，不影响最终结果，只是差一个常数因子

        def derivative(j, v, dvdj):
            # 计算导数，j可以不是整数
            v_der = dvdj
            coef = 1 / 4 + self.r_p**2 * np.exp(2 * self.delta * j) * 2 * (
                self.V_eff(j) - E
            )  # 注意ppt里面的公式漏了个2，因为notation好像不一样
            dvdj_der = v * self.delta**2 * coef
            return v_der, dvdj_der

        # RK4积分 v与v'同步更新
        h = -1
        for j in range(
            self.grid.config.j_max - 1, -1, -1
        ):  # 注意从倒数第二个点开始填充
            # r = self.grid.r[j]
            # # 自适应步长
            # min_step = 1e-5
            # h = -max(self.delta * min(1.0, r), min_step)

            # RK4 steps
            # k1
            k1v, k1dv = derivative(j + 1, v[j + 1], dvdj[j + 1])

            # k2
            k2v, k2dv = derivative(
                j + 0.5, v[j + 1] + 0.5 * h * k1v, dvdj[j + 1] + 0.5 * h * k1dv
            )

            # k3
            k3v, k3dv = derivative(
                j + 0.5, v[j + 1] + 0.5 * h * k2v, dvdj[j + 1] + 0.5 * h * k2dv
            )

            # k4
            k4v, k4dv = derivative(j, v[j + 1] + h * k3v, dvdj[j + 1] + h * k3dv)

            v[j] = v[j + 1] + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            dvdj[j] = dvdj[j + 1] + h / 6 * (k1dv + 2 * k2dv + 2 * k3dv + k4dv)

        # 变换回u(r)
        u = v * np.exp(self.delta * self.grid.j / 2)

        # 改进的归一化处理
        mask = np.abs(u) > 1e-15
        if not np.any(mask):
            return u  # 返回未归一化的波函数，让shooting_solve处理

        norm = np.sqrt(np.trapz(u[mask] * u[mask], self.grid.r[mask]))
        if norm > 0:
            u /= norm

        return u

    def shooting_solve(
        self, E_min: float, E_max: float, target_nodes: int
    ) -> Tuple[float, np.ndarray]:
        """打靶法求解本征值和本征函数"""

        def objective(E: float) -> float:
            if E >= 0:  # 确保能量为负，束缚态
                return max(1e3, 1e3 * E)  # 大的惩罚系数

            u = self.integrate_inward(E)
            nodes = WavefunctionTools.count_nodes(u)

            # 改进的打靶条件
            if nodes != target_nodes:
                return 1e3 * (nodes - target_nodes)

            # 使用波函数在原点附近的行为作为判据
            r_near = self.grid.r[:5]
            u_near = u[:5]

            if self.l == 0:
                # s态应该在原点处有限
                slope = np.polyfit(r_near, u_near, 1)[0]
                return slope
            else:
                # l>0态应该在原点处为零
                return u[0]

        try:
            # 如果设定的搜索范围不当，使用更大的搜索范围
            E_search_min = min(E_min, -2.0 / (2 * self.grid.config.n**2))
            E_search_max = max(E_max, -0.1 / (2 * self.grid.config.n**2))

            result = root_scalar(
                objective,
                bracket=[E_search_min, E_search_max],
                method="brentq",
                rtol=1e-8,
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

    def fd_solve(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
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
