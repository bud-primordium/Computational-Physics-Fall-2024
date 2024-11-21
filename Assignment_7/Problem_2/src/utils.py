"""径向薛定谔方程求解器的工具模块

包含网格生成、势能函数、波函数工具等基础设施。
提供配置类和理论值查询等功能支持。

Classes:
    SolverConfig: 求解器配置类
    RadialGrid: 径向网格工具类
    PotentialFunction: 势能函数类
    WavefunctionTools: 波函数工具类

Functions:
    get_theoretical_values: 获取理论参考值
"""

import numpy as np
from scipy.special import erf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    """径向网格工具类，处理非均匀网格的生成和变换"""

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


class PotentialFunction:  # 已经考虑了电子的负电荷
    """势能函数类"""

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
        """获取氢原子解析解

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
            解析波函数,无解析解则返回None
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
            (1, 0): -4.462,  # 1s
            (2, 0): -1.116,  # 2s
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
