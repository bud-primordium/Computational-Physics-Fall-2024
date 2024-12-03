"""
物理量数据类
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PhysicalQuantities:
    """物理量数据类"""

    # 基本物理量
    E: float = 0.0  # 能量
    E2: float = 0.0  # 能量平方
    M: float = 0.0  # 磁化强度
    M2: float = 0.0  # 磁化强度平方
    M4: float = 0.0  # 磁化强度四次方

    # 测量属性
    correlation: Optional[np.ndarray] = None  # 关联函数
    corr_length: float = 0.0  # 关联长度
    structure_factor: Optional[np.ndarray] = None  # 结构因子
    beta: float = 0.0  # 逆温度

    # 统计相关
    n_measurements: int = 0  # 测量次数
    E_sum: float = 0.0  # 能量累积和
    E2_sum: float = 0.0  # 能量平方累积和
    M_sum: float = 0.0  # 磁化累积和
    M2_sum: float = 0.0  # 磁化平方累积和
    M4_sum: float = 0.0  # 磁化四次方累积和

    def add_measurement(self, E: float, M: float):
        """添加一次测量结果"""
        # 更新单次测量值
        self.E = E
        self.E2 = E * E
        self.M = M
        self.M2 = M * M
        self.M4 = M * M * M * M

        # 更新累积和
        self.E_sum += E
        self.E2_sum += E * E
        self.M_sum += M
        self.M2_sum += M * M
        self.M4_sum += M * M * M * M

        # 更新计数
        self.n_measurements += 1

    def reset_measurements(self):
        """重置所有测量值，包括单次测量和累积和"""
        self.E = 0.0
        self.E2 = 0.0
        self.M = 0.0
        self.M2 = 0.0
        self.M4 = 0.0
        self.E_sum = 0.0
        self.E2_sum = 0.0
        self.M_sum = 0.0
        self.M2_sum = 0.0
        self.M4_sum = 0.0
        self.n_measurements = 0

    @property
    def E_mean(self) -> float:
        """能量平均值"""
        return self.E_sum / self.n_measurements if self.n_measurements > 0 else 0.0

    @property
    def E2_mean(self) -> float:
        """能量平方平均值"""
        return self.E2_sum / self.n_measurements if self.n_measurements > 0 else 0.0

    @property
    def M_mean(self) -> float:
        """磁化强度平均值"""
        return self.M_sum / self.n_measurements if self.n_measurements > 0 else 0.0

    @property
    def M2_mean(self) -> float:
        """磁化强度平方平均值"""
        return self.M2_sum / self.n_measurements if self.n_measurements > 0 else 0.0

    @property
    def M4_mean(self) -> float:
        """磁化强度四次方平均值"""
        return self.M4_sum / self.n_measurements if self.n_measurements > 0 else 0.0

    @property
    def specific_heat(self) -> float:
        """比热容 C = β²(⟨E²⟩ - ⟨E⟩²)"""
        if self.n_measurements > 0:
            return self.beta * self.beta * (self.E2_mean - self.E_mean**2)
        return 0.0

    @property
    def susceptibility(self) -> float:
        """磁化率 χ = β(⟨M²⟩ - ⟨M⟩²)"""
        if self.n_measurements > 0:
            return self.beta * (self.M2_mean - self.M_mean**2)
        return 0.0

    @property
    def binder_ratio(self) -> float:
        """Binder比 U = 1 - ⟨M⁴⟩/(3⟨M²⟩²)"""
        if self.n_measurements > 0 and self.M2_mean > 0:
            return 1.0 - self.M4_mean / (3.0 * self.M2_mean * self.M2_mean)
        return 0.0

    def to_dict(self) -> dict:
        """将物理量数据导出为字典格式，包括衍生物理量"""
        data = {
            # 基本物理量
            "E": self.E,
            "E2": self.E2,
            "M": self.M,
            "M2": self.M2,
            "M4": self.M4,
            # 统计量
            "n_measurements": self.n_measurements,
            "E_sum": self.E_sum,
            "E2_sum": self.E2_sum,
            "M_sum": self.M_sum,
            "M2_sum": self.M2_sum,
            "M4_sum": self.M4_sum,
            # 其他参数
            "beta": self.beta,
            "corr_length": self.corr_length,
        }

        # 处理numpy数组
        if self.correlation is not None:
            data["correlation"] = self.correlation.tolist()
        if self.structure_factor is not None:
            data["structure_factor"] = self.structure_factor.tolist()

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalQuantities":
        """从字典创建物理量对象"""
        # 处理所有可能的numpy数组属性
        for key in ["correlation", "structure_factor"]:
            if key in data and data[key] is not None:
                data[key] = np.array(data[key])
        return cls(**data)
