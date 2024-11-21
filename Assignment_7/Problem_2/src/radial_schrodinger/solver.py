"""径向薛定谔方程求解器的核心求解模块

实现了两种求解方法:
1. 打靶法(shooting): 从外向内积分,寻找满足边界条件的能量本征值
2. 有限差分法(finite difference): 构建矩阵直接求解本征值问题

主要特点:
- 采用变换坐标系统，处理波函数在原点附近的奇异性
- 使用改进的RK4方法进行数值积分
- 引入多重评价指标确保解的物理正确性
- 支持任意外部势能函数

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

    使用改进的RK4方法从外向内积分求解径向方程。通过多重评价指标
    (节点数、渐进行为、连续性等)寻找正确的能量本征值。

    Attributes
    ----------
    grid : RadialGrid
        计算使用的径向网格
    V : callable
        势能函数
    l : int
        角量子数
    delta : float
        网格变换参数
    r_p : float
        网格变换标度参数
    j : ndarray
        网格索引数组
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
        r_safe = np.maximum(r, 1e-10)  # 同时支持标量和数组
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
            """计算v和v'关于j的导数

            Parameters
            ----------
            j : float
                网格点(可以是半整数)
            v : float
                函数值
            dvdj : float
                函数导数值

            Returns
            -------
            tuple
                (v的导数, v'的导数)
            """
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
        self, E_min: float, E_max: float, target_nodes: int = None
    ) -> Tuple[float, np.ndarray]:
        """打靶法求解本征值和本征函数

        使用优化算法最小化多重目标函数:
        1. 波函数在原点附近的对数导数误差
        2. 波函数节点数与目标值的偏差
        3. 波函数在r_min处的幅值误差
        4. 波函数导数的连续性误差

        Parameters
        ----------
        E_min : float
            能量搜索下限
        E_max : float
            能量搜索上限
        target_nodes : int, optional
            目标节点数,默认为n-l-1

        Returns
        -------
        float
            能量本征值
        np.ndarray
            归一化的波函数

        Raises
        ------
        RuntimeError
            当优化算法未能收敛时抛出
        """
        if target_nodes is None:
            target_nodes = self.grid.config.n - self.l - 1

        # 定义目标函数的权重，可调
        w1 = 0.5  # 对数导数误差权重
        w2 = 5.0  # 节点数误差权重
        w3 = 1.0  # u(r_min) 幅值误差权重
        w4 = 1.0  # 连续性误差权重

        # 定义目标函数
        def objective(E: float) -> float:
            # 确保能量为负，束缚态
            if E >= 0:
                return 1e5 * (E - E_max)  # 惩罚正能量

            # 进行向内积分，得到波函数 u(r)
            u = self.integrate_inward(E)
            r = self.grid.r

            # 避免除以零，设置一个很小的 epsilon
            epsilon = r[0]

            # 1. 对数导数误差 修改为要求u'/u * r = l+1
            num_points = 5  # 在 r = r_min 附近选取的点数
            r_near = r[:num_points]
            u_near = u[:num_points]

            # 计算数值导数 du_dr，使用 numpy 的梯度函数
            du_dr = np.gradient(u_near, r_near)

            # 计算数值对数导数
            log_derivative_numeric = du_dr / u_near

            # 乘以 r_near，减少数值误差
            log_derivative_scaled = log_derivative_numeric * r_near
            # 计算误差
            log_derivative_error = np.sum((log_derivative_scaled - (self.l + 1)) ** 2)

            # 2. 节点数误差
            nodes = WavefunctionTools.count_nodes(u)
            nodes_diff = nodes - target_nodes
            nodes_error = nodes_diff**2  # 平方误差

            # 3. u(r_min) 的幅值误差
            u_epsilon = u[0]  # 在 r = r_min 处的波函数值
            u_epsilon_error = u_epsilon**2  # 平方误差

            # 4. 连续性误差
            du_dr_epsilon = du_dr[0]  # 在 r = ε 处的数值导数
            # 理论导数
            du_dr_theoretical = (self.l + 1) * u_epsilon / epsilon
            continuity_error = (du_dr_epsilon - du_dr_theoretical) ** 2

            # 总误差
            total_error = (
                w1 * log_derivative_error
                + w2 * nodes_error
                + w3 * u_epsilon_error
                + w4 * continuity_error
            )

            return total_error

        # 使用 scipy.optimize.minimize 进行优化
        from scipy.optimize import minimize

        def objective_wrapper(E_array):
            E = E_array[0]
            return objective(E)

        # 先粗优化
        E_grid = np.linspace(E_min, E_max, 10)  # 10 个能量点的网格
        errors = [objective(E) for E in E_grid]
        E_coarse = E_grid[np.argmin(errors)]  # 找到误差最小值对应的 E

        # 调用优化方法
        result = minimize(
            objective_wrapper,
            x0=[E_coarse],
            bounds=[(E_min, E_max)],
            method="L-BFGS-B",  # 可以尝试其他方法，如 'Nelder-Mead'
            options={"ftol": 1e-8, "disp": False, "maxls": 50},  # 增加最大线搜索次数
        )

        if result.success:
            optimal_E = result.x[0]
            # 重新计算对应的波函数
            u_optimal = self.integrate_inward(optimal_E)
            return optimal_E, u_optimal
        else:
            logger.debug(f"l-bfgs-b未收敛: {result.message},改用Nelder-Mead")
            result = minimize(
                objective_wrapper,
                x0=[E_coarse],
                bounds=[(E_min, E_max)],
                method="Nelder-Mead",
                options={"xatol": 1e-10, "disp": False, "maxiter": 1000},
            )
            if result.success:
                optimal_E = result.x[0]
                u_optimal = self.integrate_inward(optimal_E)
                return optimal_E, u_optimal
            else:
                raise RuntimeError("Nelder-Mead也未收敛")


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
        self.delta = grid.config.delta
        self.r_p = grid.r_p
        self.j = grid.j

    def V_eff(self, j) -> float:
        """计算有效势

        Parameters
        ----------
        j : array_like
            网格点索引

        Returns
        -------
        float
            有效势能
        """
        r = self.r_p * (np.exp(self.delta * j) - 1) + self.grid.config.r_min
        r_safe = np.maximum(r, 1e-10)
        return self.V(r_safe) + self.l * (self.l + 1) / (2 * r_safe * r_safe)

    def construct_hamiltonian(self):
        """构建哈密顿量矩阵

        对应变换后的径向方程:
        -[v''(j) - v(j)δ²/4] + 2δ²rp²e^(2δj)v(j)V_eff(j) = E[2δ²rp²e^(2δj)]v(j)
        从j=1开始构建方程

        Returns
        -------
        scipy.sparse.csr_matrix
            哈密顿量稀疏矩阵
        """
        N = len(self.j) - 1  # jmax+1个点，去掉最后一个点
        N_reduced = N - 1  # 再去掉第一个点
        H = lil_matrix((N_reduced, N_reduced))

        # 从j=1开始构建矩阵
        for i in range(N_reduced):
            j_actual = i + 1  # 实际的j索引
            exp_factor = (
                2
                * self.delta**2
                * self.r_p**2
                * np.exp(2 * self.delta * self.j[j_actual])
            )

            # 动能项系数
            if i > 0:  # j-1项
                H[i, i - 1] = -1
            H[i, i] = 2 + self.delta**2 / 4  # j项
            if i < N_reduced - 1:  # j+1项
                H[i, i + 1] = -1

            # 势能项
            H[i, i] += exp_factor * self.V_eff(self.j[j_actual])

        return H.tocsr()

    def construct_B_matrix(self, N_reduced):
        B = lil_matrix((N_reduced, N_reduced))
        for i in range(N_reduced):
            j_actual = i + 1
            B[i, i] = (
                2
                * self.delta**2
                * self.r_p**2
                * np.exp(2 * self.delta * self.j[j_actual])
            )
        return B.tocsr()

    def fd_solve(self, n_states: int) -> tuple:
        """渐进式求解本征值问题

        先求解最低本征值，然后根据1/n²规律估计后续本征值位置

        Parameters
        ----------
        n_states : int
            需要求解的本征态数量

        Returns
        -------
        tuple
            (energies, states)
        """
        H = self.construct_hamiltonian()
        N_reduced = H.shape[0]
        B = self.construct_B_matrix(N_reduced)

        # 首先求解最低本征值
        try:
            e_ground, v_ground = linalg.eigsh(
                H, k=1, M=B, which="SA", maxiter=500000, tol=1e-6
            )
            e_ground = e_ground[0]
        except Exception as e:
            raise RuntimeError(f"基态求解失败: {str(e)}")

        energies = [e_ground]
        states = [v_ground]

        # 使用找到的基态能量来估计后续本征值
        if n_states > 1:
            for n in range(2, n_states + 1):
                # 根据1/n²规律估计下一个本征值
                # 假设E_n = E_1/n²
                estimated_e = e_ground / (n * n)

                # 在估计值附近搜索，使用一个适当的窗口
                window = abs(e_ground) * 0.1  # 搜索窗口可以调整

                try:
                    # 在估计值附近搜索，避免找到已经找到的本征值
                    for shift in [
                        estimated_e,
                        estimated_e + window,
                        estimated_e - window,
                    ]:
                        e, v = linalg.eigsh(
                            H,
                            k=1,
                            M=B,
                            sigma=shift,
                            which="LM",
                            maxiter=10000,
                            tol=1e-7,
                        )

                        # 检查是否是新的本征值（与已有值不同）
                        if all(abs(e[0] - prev_e) > 1e-6 for prev_e in energies):
                            energies.append(e[0])
                            states.append(v)
                            break

                except Exception as e:
                    print(f"激发态 {n}求解失败: {str(e)}")
                    continue

        # 合并结果并排序
        if len(energies) > 0:
            energies = np.array(energies)
            v_states = np.hstack(states)

            # 排序
            idx = np.argsort(energies)
            energies = energies[idx]
            v_states = v_states[:, idx]

            # 转换回u并添加边界点
            u_states = np.zeros((len(self.j), len(energies)))
            for i in range(len(energies)):
                v_full = np.zeros(len(self.j))
                v_full[1:-1] = v_states[:, i]
                u_states[:, i] = v_full * np.exp(self.delta * self.j / 2)

                # 归一化
                norm = np.sqrt(np.trapz(u_states[:, i] * u_states[:, i], self.grid.r))
                if norm > 0:
                    u_states[:, i] /= norm

            return energies, u_states
        else:
            logger.error(f"本征值求解失败: {str(e)}")
            raise
