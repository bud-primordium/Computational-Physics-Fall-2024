"""径向薛定谔方程数值求解器 (Radial Schrödinger Equation Solver)

提供基于打靶法和有限差分法的径向薛定谔方程求解框架。
支持氢原子和锂原子（赝势）的能量本征值和波函数计算。

功能特点:
1. 求解方法
   - 打靶法：改进的RK4方法和多重评价指标优化
   - 有限差分法：稀疏矩阵本征值问题求解
   
2. 数值处理
   - 非均匀径向网格：r = r_p[exp(δj) - 1] + r_min
   - 波函数变换：u(r) = rR(r)
   - 原点附近奇异性处理
   - 自动波函数归一化与验证
   
3. 分析功能
   - 波函数渐近行为分析
   - 与理论值比较
   - 网格收敛性研究
   
4. 可视化
   - 波函数和概率密度分布
   - 解析解对比
   - 收敛性分析曲线

物理单位: 
    能量: Hartree 
    长度: Bohr (a)

使用示例：
    python radial_schrodinger.py --help  # 运行帮助信息
    python radial_schrodinger.py --example  # 运行内置示例
    python radial_schrodinger.py --V-type hydrogen --n 1 --l 0  # 计算氢原子基态
    python radial_schrodinger.py --convergence  # 进行收敛性分析

Author: Gilbert Young
Date: 2024-11-21
"""

# =============================================================================
# 导入模块
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from scipy.special import erf
from scipy.sparse import linalg, lil_matrix
from scipy.optimize import curve_fit

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# 工具模块
"""径向薛定谔方程求解器的工具模块

包含网格生成、势能函数、波函数工具等基础设施。
提供配置类和理论值查询等功能支持。

默认单位:
- 长度: a bohr半径
- 能量: hatree

Classes:
    SolverConfig: 求解器配置类  
    RadialGrid: 径向网格工具类
    PotentialFunction: 势能函数类
    WavefunctionTools: 波函数工具类

Functions:
    get_theoretical_values: 获取理论参考值
    get_energy_bounds: 估计能量搜索范围
"""


@dataclass
class SolverConfig:
    """求解器配置类

    Attributes
    ----------
    r_max : float
        最大径向距离 (Bohr)
    r_min : float
        最小径向距离 (Bohr)
    j_max : int
        径向网格点数
    delta : float
        网格变换参数
    l : int
        角量子数
    n : int
        主量子数
    n_states : int
        需要求解的本征态数量
    V_type : str
        势能类型 ('hydrogen' 或 'lithium')
    method : str
        求解方法 ('shooting' 或 'fd')
    tol : float
        收敛精度
    """

    r_max: float = 30.0
    r_min: float = 1e-5
    j_max: Optional[int] = None  # 现在是可选的，由方法决定默认值
    delta: Optional[float] = None  # 由j_max计算得到
    l: int = 0
    n: int = 1
    n_states: int = 3
    V_type: str = "hydrogen"
    method: str = "shooting"
    tol: float = 1e-8

    def __post_init__(self):
        """配置初始化后的验证和处理"""
        # 验证输入参数
        self._validate_inputs()

        # 设置默认网格点数
        if self.j_max is None:
            self.j_max = self._get_default_j_max()

        # 设置网格变换参数
        if self.delta is None:
            self.delta = 6 / self.j_max

    def _validate_inputs(self):
        """验证输入参数的有效性"""
        # 验证量子数
        if self.n <= 0:
            raise ValueError("主量子数n必须为正整数")
        if self.l < 0:
            raise ValueError("角量子数l必须为非负整数")
        if self.l >= self.n:
            raise ValueError("角量子数l必须小于主量子数n")

        # 验证势能类型
        if self.V_type not in ["hydrogen", "lithium"]:
            raise ValueError("不支持的势能类型")

        # 验证求解方法
        if self.method not in ["shooting", "fd"]:
            raise ValueError("不支持的求解方法")

        # 验证物理参数
        if self.r_max <= self.r_min:
            raise ValueError("r_max必须大于r_min")
        if self.r_min <= 0:
            raise ValueError("r_min必须为正数")

        # 如果手动指定了j_max，验证其范围
        if self.j_max is not None:
            if self.method == "fd" and self.j_max > 640:
                raise ValueError("有限差分法的j_max不能超过640")
            if self.j_max < 50:
                raise ValueError("网格点数过少，建议至少50个点")

    def _get_default_j_max(self) -> int:
        """根据求解方法返回默认的网格点数"""
        if self.method == "shooting":
            return 1000  # 打靶法默认使用1000个点
        else:
            return 300  # 有限差分法默认使用300个点

    @property
    def config_summary(self) -> str:
        """返回配置摘要信息"""
        return (
            f"求解配置:\n"
            f"  原子类型: {self.V_type}\n"
            f"  量子数: n={self.n}, l={self.l}\n"
            f"  求解方法: {self.method}\n"
            f"  网格点数: {self.j_max}\n"
            f"  网格范围: [{self.r_min:.1e}, {self.r_max:.1f}] Bohr"
        )


class RadialGrid:
    """径向网格工具类，处理非均匀网格的生成和变换

    使用变换 r = r_p[exp(j*delta) - 1] + r_min 生成非均匀网格,
    在原点附近较密,远处较疏,更适合原子波函数的数值计算。

    Attributes
    ----------
    config : SolverConfig
        网格配置对象
    j : ndarray
        均匀网格点索引
    r_p : float
        网格变换标度参数
    r : ndarray
        物理空间径向坐标
    dr : ndarray
        网格间距
    """

    def __init__(self, config: SolverConfig):
        """初始化网格"""
        self.config = config
        self.setup_grid()

    def setup_grid(self):
        """设置非均匀网格，使用变换 r = r_p[exp(j*delta) - 1] + r_min"""
        self.j = np.arange(self.config.j_max + 1)
        self.r_p = (self.config.r_max - self.config.r_min) / (
            np.exp(self.config.delta * self.config.j_max) - 1
        )
        self.r = self.r_p * (np.exp(self.config.delta * self.j) - 1) + self.config.r_min
        self.dr = np.diff(self.r)

    def get_grid_info(self) -> dict:
        """返回网格信息"""
        return {
            "r": self.r,
            "dr": self.dr,
            "r_min": self.r[0],
            "r_max": self.r[-1],
            "n_points": len(self.r),
        }


class PotentialFunction:
    """势能函数类

    提供不同原子的势能函数实现:
    1. 氢原子: 库仑势 V(r) = -1/r
    2. 锂原子: GTH赝势,包含局域和非局域部分
       V(r) = V_loc(r) + V_nl(r)

    注: 已考虑电子负电荷,势能直接给出
    """

    @staticmethod
    def V_hydrogen(r: np.ndarray) -> np.ndarray:
        """氢原子势能: V(r) = -1/r

        Parameters
        ----------
        r : np.ndarray
            径向距离数组

        Returns
        -------
        np.ndarray
            势能数组
        """
        r_safe = np.where(r < 1e-10, 1e-10, r)
        return -1 / r_safe

    @staticmethod
    def V_lithium(
        r: np.ndarray,
        Z_ion: float = 3,
        r_loc: float = 0.4,
        C1: float = -14.0093922,
        C2: float = 9.5099073,
        C3: float = -1.7532723,
        C4: float = 0.0834586,
    ) -> np.ndarray:
        """锂原子赝势

        Parameters
        ----------
        r : np.ndarray
            径向距离数组
        Z_ion : float, optional
            有效核电荷
        r_loc : float, optional
            局域化参数
        C1-C4 : float, optional
            势能参数

        Returns
        -------
        np.ndarray
            势能数组
        """
        r_safe = np.where(r < 1e-10, 1e-10, r)
        # 库仑项
        term1 = -Z_ion / r_safe * erf(r_safe / (np.sqrt(2) * r_loc))
        # 局域项
        exp_term = np.exp(-0.5 * (r_safe / r_loc) ** 2)
        term2 = exp_term * (
            C1
            + C2 * (r_safe / r_loc) ** 2
            + C3 * (r_safe / r_loc) ** 4
            + C4 * (r_safe / r_loc) ** 6
        )
        return term1 + term2

    @classmethod
    def get_potential(cls, V_type: str):
        """获取势能函数

        Parameters
        ----------
        V_type : str
            势能类型('hydrogen'或'lithium')

        Returns
        -------
        callable
            对应的势能函数
        """
        if V_type == "hydrogen":
            return cls.V_hydrogen
        elif V_type == "lithium":
            return cls.V_lithium
        else:
            raise ValueError(f"未知的势能类型: {V_type}")


class WavefunctionTools:
    """波函数工具类"""

    @staticmethod
    def get_analytic_hydrogen(r: np.ndarray, n: int, l: int) -> Optional[np.ndarray]:
        """获取氢原子解析解R(r)

        实现前几个低量子数态的径向波函数解析表达式:
        R(r) = N_nl * r^l * L_n^(2l+1)(2r/n) * exp(-r/n)
        其中N_nl为归一化系数,L_n^k为拉盖尔多项式

        Parameters
        ----------
        r : np.ndarray
            径向距离数组
        n : int
            主量子数
        l : int
            角量子数

        Returns
        -------
        Optional[np.ndarray]
            解析波函数,若无解析表达式则返回None
        """
        if n == 1 and l == 0:  # 1s
            return 2 * np.exp(-r)
        elif n == 2 and l == 0:  # 2s
            return np.sqrt(2) / 4 * (2 - r) * np.exp(-r / 2)
        elif n == 2 and l == 1:  # 2p
            return np.sqrt(2) / (4 * np.sqrt(3)) * r * np.exp(-r / 2)
        elif n == 3 and l == 0:  # 3s
            return 2 / 243 * np.sqrt(3) * (27 - 18 * r + 2 * r**2) * np.exp(-r / 3)
        elif n == 3 and l == 1:  # 3p
            return 2 / 81 * np.sqrt(6) * r * (6 - r) * np.exp(-r / 3)
        return None

    @staticmethod
    def count_nodes(u: np.ndarray, eps: float = 1e-10) -> int:
        """计算波函数节点数

        Parameters
        ----------
        u : np.ndarray
            波函数数组
        eps : float, optional
            判定零点的阈值

        Returns
        -------
        int
            节点数
        """
        u_filtered = np.where(abs(u) < eps, 0, u)
        return len(np.where(np.diff(np.signbit(u_filtered)))[0])


def get_theoretical_values() -> Dict:
    """获取理论参考值(Hartree单位)

    Returns
    -------
    Dict
        不同原子的能量本征值字典
        格式: {原子类型: {(n,l): 能量}}
    """
    return {
        "hydrogen": {
            (1, 0): -0.5,  # 1s
            (2, 0): -0.125,  # 2s
            (2, 1): -0.125,  # 2p
            (3, 0): -1 / 18,  # 3s
            (3, 1): -1 / 18,  # 3p
        },
        "lithium": {
            (1, 0): -4.45824675547,  # 1s
            # 跑了无数次收敛性分析出来的结果
            (2, 0): -1.115432223,  # 2s
            (2, 1): -1.12227869831,  # 2p
        },
    }


def get_energy_bounds(V_type: str, n: int) -> Tuple[float, float]:
    """获取不同原子的能量范围估计

    Parameters
    ----------
    V_type : str
        势能类型
    n : int
        主量子数

    Returns
    -------
    Tuple[float, float]
        (E_min, E_max) 能量范围估计
    """
    if V_type == "hydrogen":
        # 氢原子：E = -0.5/n^2 (精确值)
        E_center = -0.5 / (n * n)
        # 在精确值附近留出足够余量
        return E_center * 1.3, E_center * 0.7

    elif V_type == "lithium":
        # 锂原子：基态约为 -4.5，高激发态近似 -4.5/n^2
        E_center = -4.5 / (n * n)
        # 由于赝势的复杂性，留出更大的余量
        return E_center * 1.5, E_center * 0.5

    else:
        raise ValueError(f"未知的原子类型: {V_type}")


# =============================================================================
# 求解器模块
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


# =============================================================================
# 分析模块
"""径向薛定谔方程求解器的分析模块

包含波函数处理、能量分析和收敛性研究等功能。提供数值结果的验证、分析和后处理。

主要功能：
- 波函数的归一化和导数计算
- 渐近行为分析和小r极限处理 
- 能量本征值与理论值比较
- 数值方法的收敛性研究


Classes:
   WavefunctionProcessor: 波函数处理器，处理归一化和导数
   EnergyAnalyzer: 能量分析器，比较计算值和理论值
   ConvergenceAnalyzer: 收敛性分析器，研究网格依赖性
"""


class WavefunctionProcessor:
    """波函数处理类(非均匀网格版本)

    提供波函数的归一化、导数计算和渐近行为分析等功能。
    特别处理了原点附近的奇异性问题。

    Attributes
    ----------
    r : np.ndarray
        非均匀径向网格点
    l : int
        角量子数
    delta : float
        网格变换参数
    j : np.ndarray
        均匀网格点索引
    r_p : float
        网格变换参数
    dr_dj : np.ndarray
        网格变换的导数
    """

    def __init__(self, r: np.ndarray, l: int, delta: float):
        """初始化

        Parameters
        ----------
        r : np.ndarray
            非均匀径向网格点
        l : int
            角量子数
        delta : float
            网格变换参数
        """
        self.r = r
        self.l = l
        self.delta = delta
        self.j = np.arange(len(r))  # 均匀的j网格
        self.r_p = r[-1] / (np.exp(delta * (len(r) - 1)) - 1)

        # 计算网格变换的导数
        self.dr_dj = self.r_p * delta * np.exp(delta * self.j)

    def get_derivatives(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算波函数在非均匀网格上的导数

        Parameters
        ----------
        u : np.ndarray
            波函数值

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            一阶和二阶导数
        """
        du_dr = np.gradient(u, self.r)
        d2u_dr2 = np.gradient(du_dr, self.r)
        return du_dr, d2u_dr2

    def analyze_asymptotic(
        self, u: np.ndarray, num_points: int = 10
    ) -> Tuple[float, float]:
        """分析r→0时波函数的渐进行为

        通过对数拟合确定渐进形式：u(r) ~ r^m * exp(-r/n)

        Parameters
        ----------
        u : np.ndarray
            波函数值
        num_points : int
            用于拟合的点数

        Returns
        -------
        Tuple[float, float]
            (幂次m, 指数参数n)
            m应接近l+1, n应接近主量子数
        """
        # 选取近原点的几个点
        j_near_zero = self.j[:num_points]
        r_near_zero = self.r[:num_points]
        u_near_zero = u[:num_points]

        # 对数拟合
        mask = (r_near_zero > 0) & (np.abs(u_near_zero) > 1e-15)
        if not np.any(mask):
            return self.l, 1.0  # 默认值改为l

        x = np.log(r_near_zero[mask])
        y = np.log(np.abs(u_near_zero[mask]))

        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        # 计算n_analyze (从c中获取)
        n_analyze = np.exp(c)

        return m, n_analyze

    def get_r0_values(self, u: np.ndarray) -> Tuple[float, float]:
        """获取r=0处的函数值和导数"""
        # 可能弃用，因为后面绘制的是r=r_min开始的
        if self.l == 0:
            # l=0时,外推u(0),du/dr(0)=0
            du_dr, _ = self.get_derivatives(u)
            u0 = u[0] - self.r[0] * du_dr[0]  # 一阶泰勒展开
            return u0, 0.0
        else:
            # l>0时,u(0)=0,通过渐进行为确定du/dr(0)
            coef, power = self.analyze_asymptotic(u)
            if abs(power - (self.l + 1)) > 0.5:
                logger.warning(f"渐进指数{power:.2f}与预期值{self.l+1}差异较大")
            return 0.0, coef * (self.l + 1)

    def normalize_wavefunction(
        self, u: np.ndarray, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """归一化波函数并计算物理波函数R(r)

        1. 通过拟合处理r→0处的行为
        2. 归一化变换后的波函数u(r)
        3. 计算物理波函数R(r) = u(r)/r
        4. 验证归一化条件∫|R(r)|²r²dr = 1

        Parameters
        ----------
        u : np.ndarray
            输入波函数u(r)
        tol : float, optional
            归一化精度要求

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (归一化的u(r), 对应的R(r))

        Raises
        ------
        ValueError
            当波函数无效或归一化失败时
        """
        # 数值稳定性检查
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            logger.error("波函数包含无效值")
            raise ValueError("Invalid wavefunction values")

        # 处理r=r_min的情况
        # u0, _ = self.get_r0_values(u)

        # 拟合的函数模型：r^(l+1)
        def r_behavior_u(r, a):
            return a * r ** (self.l + 1)

        # 使用前几个点拟合
        fit_indices = slice(1, 7)  # 使用 u 第2~7个点（索引从1到6）
        r_fit = self.r[fit_indices]
        u_fit = u[fit_indices]

        # 执行拟合
        popt, _ = curve_fit(r_behavior_u, r_fit, u_fit)

        # 根据拟合结果外推 u_full[0]
        u_full = np.copy(u)
        u_full[0] = r_behavior_u(self.r[0], *popt)

        # 计算归一化常数(考虑非均匀网格的积分权重)
        mask = np.abs(u_full) > 1e-15
        if not np.any(mask):
            logger.error("波函数全为零")
            raise ValueError("Zero wavefunction")

        norm = np.sqrt(np.trapz(u_full[mask] * u_full[mask], self.r[mask]))
        if norm < 1e-15:
            logger.error("归一化常数接近零")
            raise ValueError("Normalization constant too small")

        u_norm = u_full / norm

        # 计算R(r) = u(r)/r
        R = np.zeros_like(u_norm)
        nonzero_r = self.r > 1e-10
        R[nonzero_r] = u_norm[nonzero_r] / self.r[nonzero_r]

        # 补丁，r=r_min附近的处理
        # 拟合的函数模型：r^(l)
        def r_behavior_R(r, a):
            return a * r ** (self.l)

        if self.l == 0:
            # 使用前几个点拟合
            fit_indices = slice(1, 7)  # 使用 R 第2~7个点（索引从1到6）
            r_fit = self.r[fit_indices]
            R_fit = R[fit_indices]

            # 执行拟合
            popt, _ = curve_fit(r_behavior_R, r_fit, R_fit)

            # 根据拟合结果外推 R_full[0]
            R_full = np.copy(R)
            R_full[0] = r_behavior_R(self.r[0], *popt)
        else:
            # 使用第10到第40个点进行拟合
            fit_indices = slice(10, 40)  # 使用第10到40个点（索引从10到39）
            r_fit = self.r[fit_indices]
            R_fit = R[fit_indices]

            # 执行拟合
            popt, _ = curve_fit(r_behavior_R, r_fit, R_fit)
            R_full = np.copy(R)
            # 根据拟合结果外推前9个点
            for i in range(9):
                R_full[i] = r_behavior_R(self.r[i], *popt)

        # # r=0处的处理
        # if self.l == 0:
        #     # 使用洛必达法则: lim(r→0) u(r)/r = du/dr(0)
        #     R[0] = du0_dr / norm
        # else:
        #     # 使用渐进行为
        #     coef, _ = self.analyze_asymptotic(u_norm)
        #     R[0] = coef * self.l

        self._verify_normalization(R, tol)
        return u_norm, R_full

    def _verify_normalization(self, R: np.ndarray, tol: float):
        """验证波函数归一化(考虑非均匀网格)"""
        mask = np.isfinite(R) & (np.abs(R) < 1e10)
        if not np.any(mask):
            logger.error("无有效值用于归一化检验")
            raise ValueError("No valid values for normalization check")

        # 在非均匀网格上积分
        integrand = R[mask] * R[mask] * self.r[mask] * self.r[mask]
        norm = np.trapz(integrand, self.r[mask])

        if not np.isfinite(norm):
            logger.error("归一化积分结果无效")
            raise ValueError("Invalid normalization integral")

        if not (1 - tol < norm < 1 + tol):
            logger.warning(
                f"归一化检验失败: ∫|R(r)|²r²dr = {norm:.6f}，请检查边界条件r_Max"
            )


class EnergyAnalyzer:
    """能量分析类"""

    def __init__(self, theoretical_values: Dict):
        """初始化

        Parameters
        ----------
        theoretical_values : Dict
            理论值字典
        """
        self.theoretical_values = theoretical_values

    def compare_with_theory(self, E: float, V_type: str, n: int, l: int) -> dict:
        """与理论值比较

        Parameters
        ----------
        E : float
            计算得到的能量
        V_type : str
            势能类型
        n : int
            主量子数
        l : int
            角量子数

        Returns
        -------
        dict
            比较结果字典
        """
        state = (n, l)
        theory = self.theoretical_values[V_type].get(state)

        result = {
            "numerical_E": E,
            "theoretical_E": theory,
            "relative_error": None,
            "status": "unknown",
        }

        if theory is not None:
            rel_error = abs((E - theory) / theory) * 100
            result.update(
                {
                    "relative_error": rel_error,
                    "status": "good" if rel_error < 1.0 else "warning",
                }
            )

        return result


class ConvergenceAnalyzer:
    """收敛性分析类"""

    def __init__(self, energy_analyzer: EnergyAnalyzer):
        """初始化

        Parameters
        ----------
        energy_analyzer : EnergyAnalyzer
            能量分析器实例
        """
        self.energy_analyzer = energy_analyzer

    def analyze_grid_convergence(self, solver, n_values: list) -> dict:
        """分析不同网格点数的收敛性

        对一系列网格点数计算能量本征值，分析相对误差随
        网格间距的变化关系，用于确定数值方法的收敛阶数。

        Parameters
        ----------
        solver : object
            求解器实例
        n_values : list
            要测试的网格点数列表

        Returns
        -------
        dict
            包含以下键的字典：
            - n_points: 网格点数列表
            - energies: 对应的能量值
            - errors: 相对误差(%)
            - delta_h: 网格间距
        """
        results = {"n_points": n_values, "energies": [], "errors": [], "delta_h": []}

        for n in n_values:
            try:
                # 使用新的网格点数求解
                E = solver.solve_with_points(n)

                # 计算误差
                analysis = self.energy_analyzer.compare_with_theory(
                    E, solver.config.V_type, solver.config.n, solver.config.l
                )

                results["energies"].append(E)
                if analysis["relative_error"] is not None:
                    results["errors"].append(analysis["relative_error"])
                results["delta_h"].append(6 / n)  # r=rp(e^t-1)+r_min t_max = 6

            except Exception as e:
                logger.warning(f"网格点数{n}的计算失败: {str(e)}")
                continue

        return results


# =============================================================================
# 可视化模块
"""径向薛定谔方程求解器的可视化模块

负责绘制波函数、概率密度、能量扫描结果等图像。
提供结果可视化和分析展示功能。

Classes:
   ResultVisualizer: 结果可视化类
"""


class ResultVisualizer:
    """结果可视化类

    提供原子波函数、能量和收敛性分析的可视化功能。
    自动处理中文显示和样式优化。

    Attributes
    ----------
    r : np.ndarray
        径向网格点
    """

    def __init__(self, r: np.ndarray):
        """初始化可视化器

        Parameters
        ----------
        r : np.ndarray
            径向网格点
        """
        self.r = r
        self._setup_style()

    def _setup_style(self):
        """设置绘图样式"""
        plt.style.use("default")

        # 尝试使用seaborn提升样式
        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid", font_scale=1.2)
        except ImportError:
            logger.info("未能导入seaborn包，使用matplotlib基本样式")

        # 检测系统
        import platform

        system = platform.system()

        # 根据操作系统设置中文字体
        if system == "Darwin":  # macOS
            plt.rcParams["font.family"] = ["Arial Unicode MS"]
        elif system == "Windows":
            plt.rcParams["font.family"] = ["Microsoft YaHei"]
        elif system == "Linux":
            plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]

        # 如果上述字体都不可用，尝试使用系统默认字体
        try:
            plt.rcParams["font.sans-serif"] = [
                "Arial Unicode MS",
                "SimSun",
                "STSong",
                "SimHei",
            ] + plt.rcParams["font.sans-serif"]
        except:
            logger.warning("未能设置理想的中文字体，尝试使用系统默认字体")

        # 设置其他参数
        plt.rcParams.update(
            {
                "figure.figsize": [10.0, 6.0],
                "figure.dpi": 100,
                "savefig.dpi": 100,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "axes.unicode_minus": False,  # 解决负号显示问题
            }
        )

    def plot_wavefunction(
        self,
        u: np.ndarray,
        R: np.ndarray,
        E: float,
        n: int,
        l: int,
        V_type: str,
        R_analytic: Optional[np.ndarray] = None,
        method: str = "shooting",
    ):
        """绘制波函数及其概率密度分布

        生成两个子图:
        1. 波函数图：展示变换后的u(r)和物理波函数R(r)
        2. 概率密度图：展示r²R²(r)分布

        当提供解析解时，同时绘制对比曲线。

        Parameters
        ----------
        u : np.ndarray
            变换坐标下的径向波函数u(r)
        R : np.ndarray
            物理坐标下的径向波函数R(r)
        E : float
            能量本征值(Hartree)
        n : int
            主量子数
        l : int
            角量子数
        V_type : str
            势能类型('hydrogen'或'lithium')
        R_analytic : np.ndarray, optional
            解析波函数(对氢原子部分态可用)
        method : str
            求解方法('shooting'或'fd')
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 上图：波函数
        ax1.plot(self.r, R, "b-", label="R(r) 数值解")
        if R_analytic is not None:
            ax1.plot(self.r, R_analytic, "r--", label="R(r) 解析解")
        ax1.plot(self.r, u, "g:", label="u(r) 数值解")
        ax1.set_xlabel("r (Bohr)")
        ax1.set_ylabel("波函数")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 下图：概率密度
        probability = R * R * self.r * self.r
        ax2.plot(self.r, probability, "b-", label="概率密度 r²R²(r)")
        if R_analytic is not None:
            prob_analytic = R_analytic * R_analytic * self.r * self.r
            ax2.plot(self.r, prob_analytic, "r--", label="解析解概率密度")
        ax2.set_xlabel("r (Bohr)")
        ax2.set_ylabel("概率密度")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 增强标题信息
        method_name = "打靶法" if method == "shooting" else "有限差分法"
        title = (
            f"{V_type.capitalize()}原子 ({method_name})\n"
            f"量子态: n={n}, l={l} | 能量: E={E:.6f} Hartree"
        )
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_energy_scan(
        self, E_array: np.ndarray, u0_array: np.ndarray, n: int, l: int, V_type: str
    ):
        """绘制能量扫描结果

        Parameters
        ----------
        E_array : np.ndarray
            能量数组
        u0_array : np.ndarray
            对应的u(0)值
        n : int
            主量子数
        l : int
            角量子数
        V_type : str
            势能类型
        """
        plt.figure(figsize=(10, 6))
        plt.plot(E_array, u0_array, "b-")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("能量 (Hartree)")
        plt.ylabel("u(r=0)")
        plt.title(f"{V_type}原子能量扫描 (n={n}, l={l})")
        plt.grid(True)
        plt.show()

    def plot_convergence_study(
        self,
        results: Dict,
        V_type: str = "",
        n: int = 1,
        l: int = 0,
        method: str = "shooting",
    ):
        """绘制收敛性分析结果

        使用双对数坐标展示网格间距与相对误差的关系。
        同时绘制二阶和四阶收敛的参考线以供比较。

        Parameters
        ----------
        results : Dict
            包含 'delta_h'(网格间距)和'errors'(相对误差)的字典
        V_type : str
            势能类型
        n : int
            主量子数
        l : int
            角量子数
        method : str
            求解方法('shooting'或'fd')
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(results["delta_h"], results["errors"], "bo-", label="数值结果")

        # 添加参考线
        h = np.array(results["delta_h"])
        if len(h) > 0:  # 确保有数据点
            plt.loglog(
                h, h**2 * results["errors"][0] / h[0] ** 2, "r--", label="O(h²) 参考线"
            )
            plt.loglog(
                h, h**4 * results["errors"][0] / h[0] ** 4, "g--", label="O(h⁴) 参考线"
            )

        plt.xlabel("网格间距 h (log)")
        plt.ylabel("相对误差 % (log)")

        # 增强标题信息
        method_name = "打靶法" if method == "shooting" else "有限差分法"
        title = (
            f"收敛性分析 ({method_name})\n" f"{V_type.capitalize()}原子: n={n}, l={l}"
        )
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# =============================================================================
# 主求解器和程序入口
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
