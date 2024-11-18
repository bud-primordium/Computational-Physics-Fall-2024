"""径向薛定谔方程求解器的基础组件"""

import numpy as np
from scipy.special import erf
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging
import warnings

# 忽略数值计算警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """求解器配置类

    Parameters
    ----------
    r_max : float
        最大径向距离, 默认35.0 Bohr
    j_max : int
        网格点数, 默认1000
    delta : float
        网格变换参数, 默认0.006
    l : int
        角量子数, 默认0
    n : int
        主量子数, 默认1
    n_states : int
        求解本征态数量, 默认3
    V_type : str
        势能类型('hydrogen'或'lithium'), 默认'hydrogen'
    method : str
        求解方法('shooting'或'fd'), 默认'shooting'
    tol : float
        收敛精度, 默认1e-12
    """

    r_max: float = 35.0
    j_max: int = 1000
    delta: float = 0.006
    l: int = 0
    n: int = 1
    n_states: int = 3
    V_type: str = "hydrogen"
    method: str = "shooting"
    tol: float = 1e-12

    def validate(self):
        """验证配置参数的有效性

        Raises
        ------
        ValueError
            当参数不满足要求时抛出
        """
        if self.r_max <= 0:
            raise ValueError("r_max必须大于0")
        if self.j_max < 100:
            raise ValueError("j_max必须至少为100")
        if self.delta <= 0:
            raise ValueError("delta必须大于0")
        if self.l < 0:
            raise ValueError("l必须为非负整数")
        if self.n <= self.l:
            raise ValueError("n必须大于l")
        if self.V_type not in ["hydrogen", "lithium"]:
            raise ValueError("不支持的势能类型")
        if self.method not in ["shooting", "fd"]:
            raise ValueError("不支持的求解方法")


class RadialGrid:
    """径向网格类

    使用变换 r = r_p[exp(j*delta) - 1] 生成非均匀网格,
    在原子核附近网格密集, 远处稀疏。

    Parameters
    ----------
    config : SolverConfig
        求解器配置对象

    Attributes
    ----------
    r : ndarray
        径向网格点
    dr : ndarray
        网格间距
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.setup_grid()

    def setup_grid(self):
        """生成非均匀网格"""
        self.j = np.arange(self.config.j_max + 1)
        self.r_p = self.config.r_max / (
            np.exp(self.config.delta * self.config.j_max) - 1
        )
        self.r = self.r_p * (np.exp(self.config.delta * self.j) - 1)
        self.dr = np.diff(self.r)

    def get_grid_info(self) -> Dict:
        """返回网格信息

        Returns
        -------
        dict
            包含网格点、间距等信息的字典
        """
        return {
            "r": self.r,
            "dr": self.dr,
            "r_min": self.r[0],
            "r_max": self.r[-1],
            "n_points": len(self.r),
        }


def get_theoretical_values() -> Dict:
    """获取理论参考值

    Returns
    -------
    dict
        能量本征值字典, 格式为:
        {原子类型: {(n,l): 能量值}}
        其中能量单位为Hartree
    """
    return {
        "hydrogen": {
            (1, 0): -0.5,  # 1s
            (2, 0): -0.125,  # 2s
            (2, 1): -0.125,  # 2p
            (3, 0): -0.0555556,  # 3s
            (3, 1): -0.0555556,  # 3p
            (3, 2): -0.0555556,  # 3d
        },
        "lithium": {
            (1, 0): -2.4776,  # 1s
            (2, 0): -0.1963,  # 2s
            (2, 1): -0.1302,  # 2p
        },
    }


class PotentialFunction:
    """势能函数类

    实现两种势能:
        - 氢原子的库仑势
        - 锂原子的局域势
    """

    @staticmethod
    def V_hydrogen(r: np.ndarray) -> np.ndarray:
        """氢原子势能

        Parameters
        ----------
        r : ndarray
            径向距离数组

        Returns
        -------
        ndarray
            势能值 V(r) = -1/r
        """
        return -1 / r

    @staticmethod
    def V_lithium(
        r: np.ndarray,
        Z_ion: float = 3.0,
        r_loc: float = 0.4,
        C1: float = -14.0093922,
        C2: float = 9.5099073,
        C3: float = -1.7532723,
        C4: float = 0.0834586,
    ) -> np.ndarray:
        """锂原子局域势

        Parameters
        ----------
        r : ndarray
            径向距离数组
        Z_ion : float, optional
            有效核电荷, 默认为3.0
        r_loc : float, optional
            局域化参数, 默认为0.4
        C1-C4 : float, optional
            势能展开系数

        Returns
        -------
        ndarray
            势能值数组

        Notes
        -----
        势能由两项组成:
        - 库仑项: -Z_ion/r * erf(r/sqrt(2)/r_loc)
        - 局域项: exp(-r²/2r_loc²)[C₁ + C₂(r/r_loc)² + ...]
        """
        term1 = -Z_ion / r * erf(r / (np.sqrt(2) * r_loc))
        exp_term = np.exp(-0.5 * (r / r_loc) ** 2)
        term2 = exp_term * (
            C1 + C2 * (r / r_loc) ** 2 + C3 * (r / r_loc) ** 4 + C4 * (r / r_loc) ** 6
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

        Raises
        ------
        ValueError
            不支持的势能类型
        """
        if V_type == "hydrogen":
            return cls.V_hydrogen
        elif V_type == "lithium":
            return cls.V_lithium
        else:
            raise ValueError(f"未知的势能类型: {V_type}")


def get_initial_energy_range(config: SolverConfig) -> Tuple[float, float]:
    """获取能量搜索范围

    Parameters
    ----------
    config : SolverConfig
        求解器配置

    Returns
    -------
    tuple of float
        (E_min, E_max) 能量搜索范围
    """
    if config.V_type == "hydrogen":
        # 氢原子能量: E_n = -1/(2n²)
        E_theo = -1.0 / (2 * config.n**2)
        E_min = E_theo * 1.5
        E_max = E_theo * 0.5
    else:
        # 锂原子能量估计
        E_theo = -3.0 / config.n**2
        E_min = E_theo * 1.5
        E_max = E_theo * 0.5

    return E_min, E_max
