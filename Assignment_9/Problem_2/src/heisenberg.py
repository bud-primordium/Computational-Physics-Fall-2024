"""
3D FCC Heisenberg Model Simulation
包含多种更新算法、FSS分析和可视化
"""

# 数值计算与优化
import numpy as np
from scipy.optimize import curve_fit
from scipy import fft

# 并行处理
import multiprocessing as mp

# 数据类和类型注解
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass

# 绘图与可视化
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import RadioButtons, Button, Slider
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.colors as mcolors

# 色彩相关
import colorsys

# 进度条
from tqdm import tqdm

# 系统和平台信息
import platform
import sys

# PyQt5 GUI库
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QMessageBox,
    QTabWidget,
    QGridLayout,
    QSpinBox,
    QDoubleSpinBox,
)


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


class HeisenbergFCC:
    """3D FCC海森堡模型的核心类"""

    def __init__(self, L: int, T: float, J: float = 1.0):
        """
        初始化FCC海森堡模型

        参数：
            L (int): 系统大小（沿每个方向的原胞数）
            T (float): 温度
            J (float): 耦合常数（默认为1.0）
        """
        self.L = L
        self.T = T
        self.J = J
        self.beta = 1.0 / T if T > 0 else float("inf")

        # 系统中的总自旋数
        self.N = 4 * L**3

        # 初始化自旋构型
        self.spins = self._init_spins()

        # 建立邻居表
        self.neighbors = self._build_neighbor_table()

        # 当前总能量
        self.energy = self.calculate_total_energy()

    def _init_spins(self) -> np.ndarray:
        """
        初始化自旋构型

        返回：
            spins (np.ndarray): 形状为(N, 3)的数组，每个位置包含一个归一化的三维自旋向量
        """
        # 随机初始化
        spins = np.random.randn(self.N, 3)
        # 归一化
        norms = np.linalg.norm(spins, axis=1, keepdims=True)
        spins = spins / norms
        return spins

    def _build_neighbor_table(self) -> List[List[int]]:
        """
        构建FCC晶格的邻居表

        返回：
            neighbors (List[List[int]]): 邻居表，neighbors[i]是与自旋i相邻的自旋索引列表
        """
        neighbors = [[] for _ in range(self.N)]

        # FCC晶格的4个基矢位置
        basis = np.array(
            [
                [0.0, 0.0, 0.0],  # 原点
                [0.0, 0.5, 0.5],  # 面心1
                [0.5, 0.0, 0.5],  # 面心2
                [0.5, 0.5, 0.0],  # 面心3
            ]
        )

        # 预定义FCC晶格的12个最近邻相对偏移
        relative_offsets = np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.0, -0.5, 0.5],
                [-0.5, 0.0, 0.5],
                [-0.5, -0.5, 0.0],
                [0.0, 0.5, -0.5],
                [0.5, 0.0, -0.5],
                [0.5, -0.5, 0.0],
                [0.0, -0.5, -0.5],
                [-0.5, 0.0, -0.5],
                [-0.5, 0.5, 0.0],
            ]
        )

        # 创建索引到坐标的映射和反映射
        index_to_coord = {}
        coord_to_index = {}

        idx = 0
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for b in range(4):
                        coord = (x + basis[b][0], y + basis[b][1], z + basis[b][2])
                        # 使用周期性边界条件，确保坐标在[0, L)范围内
                        coord = (
                            coord[0] % self.L,
                            coord[1] % self.L,
                            coord[2] % self.L,
                        )
                        index_to_coord[idx] = coord
                        coord_to_index[coord] = idx
                        idx += 1

        # 构建邻居表
        for i in range(self.N):
            x_i, y_i, z_i = index_to_coord[i]
            for offset in relative_offsets:
                x_n = (x_i + offset[0]) % self.L
                y_n = (y_i + offset[1]) % self.L
                z_n = (z_i + offset[2]) % self.L
                neighbor_coord = (x_n, y_n, z_n)
                if neighbor_coord in coord_to_index:
                    neighbor_idx = coord_to_index[neighbor_coord]
                    neighbors[i].append(neighbor_idx)

        return neighbors

    def calculate_site_energy(self, i: int) -> float:
        """
        计算单个自旋的能量

        参数：
            i (int): 自旋的索引

        返回：
            energy (float): 自旋i的能量
        """
        spin_i = self.spins[i]
        energy = 0.0
        for j in self.neighbors[i]:
            spin_j = self.spins[j]
            energy -= self.J * np.dot(spin_i, spin_j)
        return energy

    def calculate_total_energy(self) -> float:
        """
        计算系统总能量

        返回：
            total_energy (float): 系统的总能量
        """
        energy = 0.0
        for i in range(self.N):
            energy += self.calculate_site_energy(i)
        return energy / 2.0  # 避免重复计数

    def calculate_magnetization(self) -> np.ndarray:
        """
        计算系统磁化强度（向量）

        返回：
            magnetization (np.ndarray): 形状为(3,)的磁化强度向量
        """
        return np.sum(self.spins, axis=0) / self.N

    def measure_correlation(self) -> np.ndarray:
        """
        测量自旋关联函数，利用FFT加速计算

        返回：
            correlation (np.ndarray): 关联函数
        """
        # 将自旋矩阵还原为四维形状：(L, L, L, 4, 3)
        spins_reshaped = self.spins.reshape(self.L, self.L, self.L, 4, 3)
        # 对每个方向进行FFT
        fft_spins = np.fft.fftn(spins_reshaped, axes=(0, 1, 2))
        # 计算自旋的自相关函数
        corr_func = np.fft.ifftn(fft_spins * np.conj(fft_spins), axes=(0, 1, 2))
        # 取实部并平均化
        corr = np.real(corr_func).mean(axis=(3, 4)) / self.N
        # 提取从原点到最大距离的关联函数
        correlation = corr.flatten()
        return correlation

    def measure_structure_factor(self) -> np.ndarray:
        """
        测量结构因子（傅里叶空间的关联函数）

        返回：
            structure_factor (np.ndarray): 形状为(L, L, L)的结构因子
        """
        # 将自旋矩阵还原为四维形状：(L, L, L, 4, 3)
        spins_reshaped = self.spins.reshape(self.L, self.L, self.L, 4, 3)
        # 对自旋进行FFT
        fft_spins = np.fft.fftn(spins_reshaped, axes=(0, 1, 2))
        # 计算结构因子
        sf = np.sum(np.abs(fft_spins) ** 2, axis=(3, 4)) / self.N
        return sf

    def estimate_correlation_length(self) -> float:
        """
        估计关联长度，使用自旋关联函数的指数衰减拟合

        返回：
            xi (float): 关联长度
        """
        corr = self.measure_correlation()
        x = np.arange(len(corr))
        # 初始猜测
        p0 = [1.0, corr[0]]

        # 定义指数衰减函数
        def exponential_decay(r, xi, A):
            return A * np.exp(-r / xi)

        # 进行非线性拟合
        try:
            params, _ = curve_fit(exponential_decay, x, corr, p0=p0)
            xi = params[0]
            return xi
        except RuntimeError:
            return 0.0

    def update_spin(self, i: int, new_spin: np.ndarray):
        """
        更新自旋i的值，并更新系统能量

        参数：
            i (int): 自旋的索引
            new_spin (np.ndarray): 新的自旋向量
        """
        delta_energy = 0.0
        spin_i_old = self.spins[i]
        for j in self.neighbors[i]:
            spin_j = self.spins[j]
            delta_energy += self.J * np.dot(spin_i_old - new_spin, spin_j)

        # 更新自旋和总能量
        self.spins[i] = new_spin
        self.energy += delta_energy

    def get_spin(self, i: int) -> np.ndarray:
        """
        获取自旋i的值

        参数：
            i (int): 自旋的索引

        返回：
            spin (np.ndarray): 自旋向量
        """
        return self.spins[i]

    def get_num_spins(self) -> int:
        """
        获取系统中的自旋总数

        返回：
            N (int): 自旋总数
        """
        return self.N

    def get_neighbors(self, i: int) -> List[int]:
        """
        获取自旋i的邻居索引列表

        参数：
            i (int): 自旋的索引

        返回：
            neighbors (List[int]): 邻居索引列表
        """
        return self.neighbors[i]

    def _get_index_to_coord(self) -> Dict[int, Tuple[float, float, float]]:
        """
        获取自旋索引到坐标的映射

        返回：
            index_to_coord (Dict[int, Tuple[float, float, float]]): 索引到坐标的映射
        """
        index_to_coord = {}
        basis = np.array(
            [
                [0.0, 0.0, 0.0],  # 原点
                [0.0, 0.5, 0.5],  # 面心1
                [0.5, 0.0, 0.5],  # 面心2
                [0.5, 0.5, 0.0],  # 面心3
            ]
        )

        idx = 0
        for x in range(self.L):
            for y in range(self.L):
                for z in range(self.L):
                    for b in range(4):
                        coord = (x + basis[b][0], y + basis[b][1], z + basis[b][2])
                        # 使用周期性边界条件，确保坐标在[0, L)范围内
                        coord = (
                            coord[0] % self.L,
                            coord[1] % self.L,
                            coord[2] % self.L,
                        )
                        index_to_coord[idx] = coord
                        idx += 1
        return index_to_coord


class UpdaterBase:
    """更新算法的基类"""

    def __init__(self, model: HeisenbergFCC):
        self.model = model
        self.beta = model.beta
        self.J = model.J

    def update(self) -> None:
        raise NotImplementedError


class SingleSpinUpdate(UpdaterBase):
    """单自旋Metropolis更新"""

    def __init__(self, model: HeisenbergFCC, delta_theta_max: float = 0.1):
        super().__init__(model)
        self.delta_theta_max = min(delta_theta_max, np.pi)  # 最大角度变化
        self.n_proposed = 0  # 提议次数
        self.n_accepted = 0  # 接受次数

    def _random_rotation(self, spin: np.ndarray) -> np.ndarray:
        """生成随机旋转后的自旋"""
        # 随机旋转角度和轴
        delta_theta = np.random.uniform(0, self.delta_theta_max)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        # Rodrigues旋转公式
        new_spin = (
            spin * np.cos(delta_theta)
            + np.cross(axis, spin) * np.sin(delta_theta)
            + axis * np.dot(axis, spin) * (1 - np.cos(delta_theta))
        )
        return new_spin / np.linalg.norm(new_spin)

    def update(self) -> None:
        """执行一次单自旋更新"""
        i = np.random.randint(0, self.model.N)
        old_spin = self.model.get_spin(i).copy()
        old_energy = self.model.calculate_site_energy(i)

        new_spin = self._random_rotation(old_spin)
        self.model.spins[i] = new_spin
        new_energy = self.model.calculate_site_energy(i)

        delta_E = new_energy - old_energy

        # Metropolis判据
        self.n_proposed += 1
        if delta_E <= 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            self.n_accepted += 1
            self.model.energy += delta_E
        else:
            self.model.spins[i] = old_spin

    def sweep(self) -> None:
        """对所有自旋进行一次完整更新"""
        for _ in range(self.model.N):
            self.update()

    @property
    def acceptance_rate(self) -> float:
        """计算接受率"""
        return self.n_accepted / self.n_proposed if self.n_proposed > 0 else 0.0

    def adjust_step_size(
        self, target_rate: float = 0.5, tolerance: float = 0.05
    ) -> None:
        """调整步长以达到目标接受率"""
        if self.n_proposed == 0:
            return
        current_rate = self.acceptance_rate
        if abs(current_rate - target_rate) > tolerance:
            self.delta_theta_max *= current_rate / target_rate
            self.delta_theta_max = min(max(self.delta_theta_max, 1e-5), np.pi)
        self.n_proposed = self.n_accepted = 0


class SwendsenWangUpdate(UpdaterBase):
    """Swendsen-Wang群集更新"""

    def __init__(self, model: HeisenbergFCC):
        super().__init__(model)
        self.labels = np.arange(model.N)  # 群集标签
        self.clusters: List[Set[int]] = []  # 群集列表
        self.projection_dir = None  # 投影方向

    def _find(self, x: int) -> int:
        """并查集查找操作"""
        if self.labels[x] != x:
            self.labels[x] = self._find(self.labels[x])
        return self.labels[x]

    def _union(self, x: int, y: int) -> None:
        """并查集合并操作"""
        x_root, y_root = self._find(x), self._find(y)
        if x_root != y_root:
            self.labels[y_root] = x_root

    def _generate_projection_direction(self) -> None:
        """生成随机投影方向"""
        dir = np.random.randn(3)
        self.projection_dir = dir / np.linalg.norm(dir)

    def update(self) -> None:
        """执行一次Swendsen-Wang群集更新"""
        # 重置并初始化
        self.labels = np.arange(self.model.N)
        self._generate_projection_direction()

        # 构建群集
        spins = self.model.spins
        r = self.projection_dir
        beta_J = self.beta * self.J

        # 遍历所有相邻对，建立连接
        for i in range(self.model.N):
            Si_proj = np.dot(spins[i], r)
            for j in self.model.neighbors[i]:
                if i < j:
                    Sj_proj = np.dot(spins[j], r)
                    if Si_proj * Sj_proj > 0:  # 投影自旋同向
                        pij = 1 - np.exp(-2 * beta_J * Si_proj * Sj_proj)
                        if np.random.rand() < pij:
                            self._union(i, j)

        # 提取群集
        for i in range(self.model.N):
            self._find(i)

        clusters_dict = {}
        for i in range(self.model.N):
            root = self.labels[i]
            if root not in clusters_dict:
                clusters_dict[root] = set()
            clusters_dict[root].add(i)
        self.clusters = list(clusters_dict.values())

        # 更新群集
        r = self.projection_dir
        for cluster in self.clusters:
            if np.random.rand() < 0.5:  # 以0.5概率翻转群集
                for i in cluster:
                    Si_proj = np.dot(spins[i], r)
                    spins[i] -= 2 * Si_proj * r

        self.model.spins = spins
        self.model.energy = self.model.calculate_total_energy()


class WolffUpdate(UpdaterBase):
    """Wolff单群集更新"""

    def __init__(self, model: HeisenbergFCC):
        super().__init__(model)
        self.cluster: Set[int] = set()  # 当前群集
        self.projection_dir = None  # 投影方向

    def _generate_projection_direction(self) -> None:
        """生成随机投影方向"""
        dir = np.random.randn(3)
        self.projection_dir = dir / np.linalg.norm(dir)

    def update(self) -> None:
        """执行一次Wolff单群集更新"""
        self._generate_projection_direction()
        start_site = np.random.randint(0, self.model.N)

        # 构建群集
        stack = [start_site]
        self.cluster = {start_site}
        spins = self.model.spins
        r = self.projection_dir

        # 深度优先搜索构建群集
        while stack:
            current = stack.pop()
            Si_proj = np.dot(spins[current], r)

            for neighbor in self.model.neighbors[current]:
                if neighbor not in self.cluster:
                    Sj_proj = np.dot(spins[neighbor], r)
                    if Si_proj * Sj_proj > 0:  # 投影自旋同向
                        prob = 1 - np.exp(-2 * self.beta * self.J * Si_proj * Sj_proj)
                        if np.random.rand() < prob:
                            stack.append(neighbor)
                            self.cluster.add(neighbor)

        # 翻转群集
        for i in self.cluster:
            Si_proj = np.dot(self.model.spins[i], r)
            self.model.spins[i] -= 2 * Si_proj * r

        self.model.energy = self.model.calculate_total_energy()


def create_updater(model: HeisenbergFCC, updater_type: str) -> UpdaterBase:
    """创建更新器的工厂函数"""
    updaters = {
        "single": SingleSpinUpdate,
        "sw": SwendsenWangUpdate,
        "wolff": WolffUpdate,
    }
    return updaters[updater_type](model)


class FSAnalysis:
    def __init__(self):
        self.beta = 0.3645  # 3D Heisenberg
        self.gamma = 1.386
        self.nu = 0.7112

    def estimate_Tc(
        self,
        L_values: List[int],
        T_values: List[float],
        binder_data: Dict[Tuple[int, float], float],
    ) -> float:
        """估计临界温度

        参数:
            L_values: 系统尺寸列表
            T_values: 温度列表
            binder_data: 字典，键为(L,T)，值为对应的Binder比
        """
        crossings = []
        for i, L1 in enumerate(L_values[:-1]):
            for L2 in L_values[i + 1 :]:
                binder1 = np.array([binder_data[(L1, T)] for T in T_values])
                binder2 = np.array([binder_data[(L2, T)] for T in T_values])

                # 找到交点
                idx = np.argmin(np.abs(binder1 - binder2))
                crossings.append(T_values[idx])

        return np.mean(crossings)

    def data_collapse(
        self,
        L_values: List[int],
        T_values: List[float],
        observable: str,
        data: Dict[Tuple[int, float], float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """数据塌缩分析"""
        Tc = self.estimate_Tc(L_values, T_values, data)
        x_scaled = []
        y_scaled = []

        for L in L_values:
            t = (T_values - Tc) / Tc
            x = L ** (1 / self.nu) * t

            y_values = np.array([data[(L, T)] for T in T_values])
            if observable == "M":
                y = y_values * L ** (self.beta / self.nu)
            else:  # χ
                y = y_values * L ** (-self.gamma / self.nu)

            x_scaled.extend(x)
            y_scaled.extend(y)

        return np.array(x_scaled), np.array(y_scaled)


class Simulation:
    def __init__(
        self,
        L: int,
        T: float,
        updater_type: str,
        mc_steps: int = 1000,
        thermalization_steps: int = 100,
    ):
        self.model = HeisenbergFCC(L=L, T=T)
        self.updater = create_updater(self.model, updater_type)
        self.mc_steps = mc_steps
        self.thermalization_steps = thermalization_steps

    def run(self) -> List[PhysicalQuantities]:
        # 热化
        for _ in range(self.thermalization_steps):
            self.updater.update()

        # 测量
        measurements = []
        for _ in range(self.mc_steps):
            self.updater.update()
            quantities = PhysicalQuantities()
            quantities.E = self.model.energy / self.model.N
            M = np.linalg.norm(self.model.calculate_magnetization())
            quantities.M = M
            quantities.beta = self.model.beta
            measurements.append(quantities)

        return measurements


def run_single_simulation(params):
    L, T, updater_type, steps, therm_steps = params
    sim = Simulation(L, T, updater_type, steps, therm_steps)
    return sim.run()


class AnnealingSimulation:
    """模拟退火类，用于研究相变

    通过逐步降温来研究系统的相变行为，并收集各个温度点的物理量
    """

    def __init__(
        self,
        L: int,
        T_start: float,
        T_end: float,
        cooling_steps: int,
        mc_steps_per_T: int = 1000,
        thermalization_steps: int = 100,
    ):
        """
        参数:
            L: 系统大小
            T_start: 起始温度
            T_end: 终止温度
            cooling_steps: 降温步数
            mc_steps_per_T: 每个温度点的MC步数
            thermalization_steps: 每个温度点的热化步数
        """
        self.L = L
        self.T_start = T_start
        self.T_end = T_end
        self.cooling_steps = cooling_steps
        self.mc_steps = mc_steps_per_T
        self.thermalization_steps = thermalization_steps

        # 保存结果
        self.T_values = np.linspace(T_start, T_end, cooling_steps)
        self.energy_values = []
        self.magnetization_values = []
        self.specific_heat_values = []
        self.susceptibility_values = []
        self.binder_values = []

    def run(self) -> Dict[str, np.ndarray]:
        """执行模拟退火并返回结果

        返回:
            包含各物理量数据的字典
        """
        # 初始化模型（从高温开始）
        model = HeisenbergFCC(L=self.L, T=self.T_start)
        results = {
            "temperature": self.T_values,
            "energy": [],
            "magnetization": [],
            "specific_heat": [],
            "susceptibility": [],
            "binder": [],
        }

        # 对每个温度点进行模拟
        for T in tqdm(self.T_values, desc=f"Annealing L={self.L}"):
            # 更新温度
            model.T = T
            model.beta = 1.0 / T

            # 创建更新器（使用Wolff算法）
            updater = create_updater(model, "wolff")

            # 热化
            for _ in range(self.thermalization_steps):
                updater.update()

            # 收集数据
            E_samples = []
            M_samples = []
            M2_samples = []
            M4_samples = []

            # 进行测量
            for _ in range(self.mc_steps):
                updater.update()
                E = model.energy / model.N
                M = np.linalg.norm(model.calculate_magnetization())

                E_samples.append(E)
                M_samples.append(M)
                M2_samples.append(M * M)
                M4_samples.append(M * M * M * M)

            # 计算平均值和误差
            E_mean = np.mean(E_samples)
            M_mean = np.mean(M_samples)
            E2_mean = np.mean([e * e for e in E_samples])
            M2_mean = np.mean(M2_samples)
            M4_mean = np.mean(M4_samples)

            # 计算物理量
            C = model.beta * model.beta * (E2_mean - E_mean * E_mean)
            chi = model.beta * (M2_mean - M_mean * M_mean)
            binder = 1.0 - M4_mean / (3.0 * M2_mean * M2_mean)

            # 存储结果
            results["energy"].append(E_mean)
            results["magnetization"].append(M_mean)
            results["specific_heat"].append(C)
            results["susceptibility"].append(chi)
            results["binder"].append(binder)

        # 转换为numpy数组
        for key in results:
            if key != "temperature":
                results[key] = np.array(results[key])

        return results

    def analyze_phase_transition(self, fss: FSAnalysis) -> Dict:
        """使用FSS分析相变

        参数:
            fss: FSAnalysis实例

        返回:
            包含分析结果的字典
        """
        results = self.run()

        # 构造Binder比数据
        binder_data = {(self.L, T): b for T, b in zip(self.T_values, results["binder"])}

        # 估计Tc
        Tc = fss.estimate_Tc([self.L], self.T_values, binder_data)

        return {"Tc": Tc, "results": results}

    def plot_results(self, results: Dict[str, np.ndarray]) -> None:
        """绘制模拟结果

        参数:
            results: run()方法返回的结果字典
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # 能量
        ax1.plot(results["temperature"], results["energy"], "o-")
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy per spin")
        ax1.grid(True)

        # 磁化强度
        ax2.plot(results["temperature"], results["magnetization"], "o-")
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Magnetization")
        ax2.grid(True)

        # 比热容
        ax3.plot(results["temperature"], results["specific_heat"], "o-")
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Specific Heat")
        ax3.grid(True)

        # 磁化率
        ax4.plot(results["temperature"], results["susceptibility"], "o-")
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


class ParallelSimulation:
    def __init__(
        self,
        L_values: List[int],
        T_values: List[float],
        updater_types: List[str],
        mc_steps: int = 1000,
        thermalization_steps: int = 100,
    ):
        self.params = [
            (L, T, u, mc_steps, thermalization_steps)
            for L in L_values
            for T in T_values
            for u in updater_types
        ]
        self.results: Dict = {}

    def run(self):
        with mp.Pool() as pool:
            results = pool.map(run_single_simulation, self.params)

        # 整理结果
        for (L, T, u, _, _), meas in zip(self.params, results):
            key = (L, T, u)
            self.results[key] = meas

    def get_observable(self, observable: str) -> Dict:
        result = {}
        for (L, T, u), measurements in self.results.items():
            values = [getattr(m, observable) for m in measurements]
            result[(L, T, u)] = np.array(values)
        return result

    def run_annealing(self, T_start: float, T_end: float, cooling_steps: int) -> Dict:
        """对不同尺寸系统进行模拟退火"""
        results = {}
        for L in self.L_values:
            sim = AnnealingSimulation(
                L=L,
                T_start=T_start,
                T_end=T_end,
                cooling_steps=cooling_steps,
                mc_steps_per_T=self.mc_steps,
                thermalization_steps=self.thermalization_steps,
            )
            results[L] = sim.run()
        return results


def configure_matplotlib_fonts():
    """配置matplotlib的字体设置"""
    system = platform.system()
    if system == "Darwin":  # macOS
        plt.rcParams["font.family"] = ["Arial Unicode MS"]
    elif system == "Windows":
        plt.rcParams["font.family"] = ["Microsoft YaHei"]
    else:  # Linux
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
    # 备用字体
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


class Visualization:
    def __init__(self):
        self.colormap = plt.cm.viridis
        # 为自旋和簇添加颜色映射
        self.spin_colors = {
            "arrows": "royalblue",
            "points": "gray",
            "clusters": self._generate_cluster_colors(20),  # 预生成20种不同的颜色
        }
        configure_matplotlib_fonts()

    def _generate_cluster_colors(self, n: int) -> List[str]:
        """生成n种视觉上易区分的颜色"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  # 交替使用不同饱和度
            value = 0.8 + 0.2 * (i % 2)  # 交替使用不同明度
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors

    def _get_cluster_colors(self, clusters: List[Set[int]]) -> Dict[int, str]:
        """为每个簇分配颜色"""
        colors = {}
        for i, cluster in enumerate(clusters):
            color = self.spin_colors["clusters"][i % len(self.spin_colors["clusters"])]
            for spin_idx in cluster:
                colors[spin_idx] = color
        return colors

    def plot_spins(
        self, model: HeisenbergFCC, clusters: Optional[List[Set[int]]] = None, ax=None
    ) -> None:
        """
        绘制自旋构型，可选择性地显示簇

        参数:
            model: HeisenbergFCC模型实例
            clusters: 可选的簇列表
            ax: 可选的 Matplotlib Axes 对象
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax.clear()

        coords = np.array([coord for coord in model._get_index_to_coord().values()])
        spins = model.spins

        # 如果提供了簇信息，使用不同颜色显示
        if clusters:
            cluster_colors = self._get_cluster_colors(clusters)
            colors = [
                cluster_colors.get(i, self.spin_colors["arrows"])
                for i in range(len(spins))
            ]
        else:
            colors = [self.spin_colors["arrows"]] * len(spins)

        # 绘制格点
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=self.spin_colors["points"],
            alpha=0.3,
        )

        # 绘制自旋箭头
        for coord, spin, color in zip(coords, spins, colors):
            ax.quiver(
                coord[0],
                coord[1],
                coord[2],
                spin[0],
                spin[1],
                spin[2],
                color=color,
                length=0.2,
                normalize=True,
                alpha=0.8,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Spin Configuration" + (" with Clusters" if clusters else ""))

        if ax is None:
            plt.show()

    def animate_update(
        self, model: HeisenbergFCC, updater, steps: int = 100
    ) -> FuncAnimation:
        """改进的动画函数，显示簇的更新"""
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 4, 1])

        title_ax = fig.add_subplot(gs[0])
        title_ax.axis("off")
        title = title_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=12)

        ax = fig.add_subplot(gs[1], projection="3d")
        coords = np.array([coord for coord in model._get_index_to_coord().values()])

        # 存储所有箭头对象
        arrows = []

        info_ax = fig.add_subplot(gs[2])
        info_ax.axis("off")
        info = info_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

        def init():
            ax.clear()
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=self.spin_colors["points"],
                alpha=0.3,
                s=50,
            )

            # 初始化箭头
            arrows.clear()
            for coord, spin in zip(coords, model.spins):
                arrow = ax.quiver(
                    coord[0],
                    coord[1],
                    coord[2],
                    spin[0],
                    spin[1],
                    spin[2],
                    color=self.spin_colors["arrows"],
                    length=0.3,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    linewidth=1.5,
                    alpha=0.8,
                )
                arrows.append(arrow)

            ax.set_xlim(-0.5, model.L + 0.5)
            ax.set_ylim(-0.5, model.L + 0.5)
            ax.set_zlim(-0.5, model.L + 0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            title.set_text(f"FCC Heisenberg Model (L={model.L}, T={model.T:.2f})")
            return arrows + [title, info]

        def update(frame):
            E_before = model.energy / model.N
            M_before = np.linalg.norm(model.calculate_magnetization())

            # 执行更新
            updater.update()

            E_after = model.energy / model.N
            M_after = np.linalg.norm(model.calculate_magnetization())

            # 清除旧箭头
            for arrow in arrows:
                arrow.remove()
            arrows.clear()

            # 获取簇信息并分配颜色
            colors = [self.spin_colors["arrows"]] * model.N
            if hasattr(updater, "cluster"):
                cluster = updater.cluster
                cluster_color = self.spin_colors["clusters"][
                    frame % len(self.spin_colors["clusters"])
                ]
                for idx in cluster:
                    colors[idx] = cluster_color
            elif hasattr(updater, "clusters"):
                cluster_colors = self._get_cluster_colors(updater.clusters)
                colors = [
                    cluster_colors.get(i, self.spin_colors["arrows"])
                    for i in range(model.N)
                ]

            # 绘制新箭头
            for coord, spin, color in zip(coords, model.spins, colors):
                arrow = ax.quiver(
                    coord[0],
                    coord[1],
                    coord[2],
                    spin[0],
                    spin[1],
                    spin[2],
                    color=color,
                    length=0.3,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    linewidth=1.5,
                    alpha=0.8,
                )
                arrows.append(arrow)

            # 更新信息显示
            info_text = (
                f"Step: {frame}\n"
                f"Energy/N: {E_after:.4f} (ΔE/N: {E_after-E_before:.4f})\n"
                f"Magnetization: {M_after:.4f} (ΔM: {M_after-M_before:.4f})"
            )

            if hasattr(updater, "cluster"):
                info_text += f"\nCluster Size: {len(updater.cluster)}"
            elif hasattr(updater, "clusters"):
                total_size = sum(len(c) for c in updater.clusters)
                info_text += f"\nNumber of Clusters: {len(updater.clusters)}"
                info_text += f"\nTotal Spins in Clusters: {total_size}"

            info.set_text(info_text)
            return arrows + [title, info]

        plt.tight_layout()

        ani = FuncAnimation(
            fig, update, frames=steps, init_func=init, interval=200, blit=False
        )
        return ani

    def plot_correlation(self, corr: np.ndarray, distance: np.ndarray) -> None:
        """绘制关联函数"""
        plt.figure(figsize=(8, 6))
        plt.plot(distance, corr, "o-")
        plt.xlabel("Distance")
        plt.ylabel("Correlation")
        plt.yscale("log")
        plt.title("Spin-Spin Correlation Function")
        plt.grid(True)
        plt.show()

    def plot_structure_factor(self, sf: np.ndarray) -> None:
        """绘制结构因子"""
        L = sf.shape[0]
        extent = [-np.pi, np.pi, -np.pi, np.pi]

        plt.figure(figsize=(8, 6))
        plt.imshow(sf[:, :, L // 2], extent=extent, cmap="hot")
        plt.colorbar(label="S(q)")
        plt.xlabel("qx")
        plt.ylabel("qy")
        plt.title("Structure Factor S(q) [qz=0 plane]")
        plt.show()

    def plot_physical_quantities(
        self, results: Dict[Tuple[int, float, str], List]
    ) -> None:
        """绘制物理量随温度的变化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        L_values = sorted(set(L for (L, _, _) in results.keys()))
        T_values = sorted(set(T for (_, T, _) in results.keys()))

        for L in L_values:
            E = [np.mean([m.E for m in results[(L, T, "wolff")]]) for T in T_values]
            M = [np.mean([m.M for m in results[(L, T, "wolff")]]) for T in T_values]
            C = [
                np.mean([m.specific_heat for m in results[(L, T, "wolff")]])
                for T in T_values
            ]
            chi = [
                np.mean([m.susceptibility for m in results[(L, T, "wolff")]])
                for T in T_values
            ]

            ax1.plot(T_values, E, "o-", label=f"L={L}")
            ax2.plot(T_values, M, "o-", label=f"L={L}")
            ax3.plot(T_values, C, "o-", label=f"L={L}")
            ax4.plot(T_values, chi, "o-", label=f"L={L}")

        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Magnetization")
        ax2.legend()
        ax2.grid(True)

        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Specific Heat")
        ax3.legend()
        ax3.grid(True)

        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class InteractiveApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Heisenberg Model Visualization")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化模型参数
        self.L = 4
        self.T = 1.0
        self.algorithm = "wolff"
        self.measurement_interval = 10
        self.current_step = 0

        # 创建模型和可视化实例
        self.model = HeisenbergFCC(L=self.L, T=self.T)
        self.updater = create_updater(self.model, self.algorithm)
        self.vis = Visualization()

        # 存储物理量历史数据
        self.energy_history = []
        self.magnetization_history = []
        self.steps = []

        # 创建主窗口部件
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # 创建标签页
        self.create_tabs()

        # 创建控制面板
        self.create_control_panel()

        # 定时器用于动画
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.animating = False

    def create_control_panel(self):
        control_panel = QWidget()
        control_layout = QGridLayout()

        # 系统参数控制
        control_layout.addWidget(QLabel("System Size (L):"), 0, 0)
        self.L_spin = QSpinBox()
        self.L_spin.setRange(2, 32)
        self.L_spin.setValue(self.L)
        control_layout.addWidget(self.L_spin, 0, 1)

        control_layout.addWidget(QLabel("Temperature (T):"), 0, 2)
        self.T_spin = QDoubleSpinBox()
        self.T_spin.setRange(0.1, 10.0)
        self.T_spin.setValue(self.T)
        self.T_spin.setSingleStep(0.1)
        control_layout.addWidget(self.T_spin, 0, 3)

        control_layout.addWidget(QLabel("Algorithm:"), 0, 4)
        self.alg_combo = QComboBox()
        self.alg_combo.addItems(["wolff", "sw", "single"])
        self.alg_combo.setCurrentText(self.algorithm)
        control_layout.addWidget(self.alg_combo, 0, 5)

        # 控制按钮
        self.update_button = QPushButton("Update Model")
        self.update_button.clicked.connect(self.update_model)
        control_layout.addWidget(self.update_button, 1, 0)

        self.animate_button = QPushButton("Start Animation")
        self.animate_button.clicked.connect(self.toggle_animation)
        control_layout.addWidget(self.animate_button, 1, 1)

        self.reset_button = QPushButton("Reset Data")
        self.reset_button.clicked.connect(self.reset_data)
        control_layout.addWidget(self.reset_button, 1, 2)

        control_panel.setLayout(control_layout)
        self.layout.addWidget(control_panel)

    def create_tabs(self):
        self.tabs = QTabWidget()

        # 自旋构型标签页
        self.spins_tab = QWidget()
        spins_layout = QVBoxLayout()
        self.spins_canvas = MplCanvas(self, width=8, height=6)
        self.spins_toolbar = NavigationToolbar(self.spins_canvas, self)
        spins_layout.addWidget(self.spins_toolbar)
        spins_layout.addWidget(self.spins_canvas)
        self.spins_tab.setLayout(spins_layout)

        # 物理量演化标签页
        self.evolution_tab = QWidget()
        evolution_layout = QVBoxLayout()
        self.evolution_canvas = MplCanvas(self, width=8, height=6)
        self.evolution_toolbar = NavigationToolbar(self.evolution_canvas, self)
        evolution_layout.addWidget(self.evolution_toolbar)
        evolution_layout.addWidget(self.evolution_canvas)
        self.evolution_tab.setLayout(evolution_layout)

        # 关联函数标签页
        self.correlation_tab = QWidget()
        correlation_layout = QVBoxLayout()
        self.correlation_canvas = MplCanvas(self, width=8, height=6)
        self.correlation_toolbar = NavigationToolbar(self.correlation_canvas, self)
        correlation_layout.addWidget(self.correlation_toolbar)
        correlation_layout.addWidget(self.correlation_canvas)
        self.correlation_tab.setLayout(correlation_layout)

        # 结构因子标签页
        self.structure_tab = QWidget()
        structure_layout = QVBoxLayout()
        self.structure_canvas = MplCanvas(self, width=8, height=6)
        self.structure_toolbar = NavigationToolbar(self.structure_canvas, self)
        structure_layout.addWidget(self.structure_toolbar)
        structure_layout.addWidget(self.structure_canvas)
        self.structure_tab.setLayout(structure_layout)

        # 添加相变分析标签页
        self.phase_transition_tab = QWidget()
        phase_layout = QVBoxLayout()

        # 控制面板
        control_panel = QWidget()
        control_grid = QGridLayout()

        # 温度范围控制
        control_grid.addWidget(QLabel("T start:"), 0, 0)
        self.T_start_spin = QDoubleSpinBox()
        self.T_start_spin.setRange(0.1, 10.0)
        self.T_start_spin.setValue(1.0)
        self.T_start_spin.setSingleStep(0.1)
        control_grid.addWidget(self.T_start_spin, 0, 1)

        control_grid.addWidget(QLabel("T end:"), 0, 2)
        self.T_end_spin = QDoubleSpinBox()
        self.T_end_spin.setRange(0.1, 10.0)
        self.T_end_spin.setValue(10.0)
        self.T_end_spin.setSingleStep(0.1)
        control_grid.addWidget(self.T_end_spin, 0, 3)

        control_grid.addWidget(QLabel("Cooling steps:"), 0, 4)
        self.cooling_steps_spin = QSpinBox()
        self.cooling_steps_spin.setRange(10, 1000)
        self.cooling_steps_spin.setValue(200)
        control_grid.addWidget(self.cooling_steps_spin, 0, 5)

        # 运行按钮
        self.run_annealing_button = QPushButton("Run Annealing")
        self.run_annealing_button.clicked.connect(self.run_phase_transition_analysis)
        control_grid.addWidget(self.run_annealing_button, 1, 0, 1, 2)

        control_panel.setLayout(control_grid)
        phase_layout.addWidget(control_panel)

        # 结果显示区域
        self.phase_canvas = MplCanvas(self, width=8, height=6)
        self.phase_toolbar = NavigationToolbar(self.phase_canvas, self)
        phase_layout.addWidget(self.phase_toolbar)
        phase_layout.addWidget(self.phase_canvas)

        self.phase_transition_tab.setLayout(phase_layout)

        # 添加标签页
        self.tabs.addTab(self.spins_tab, "Spin Configuration")
        self.tabs.addTab(self.evolution_tab, "Physical Quantities")
        self.tabs.addTab(self.correlation_tab, "Correlation Function")
        self.tabs.addTab(self.structure_tab, "Structure Factor")
        self.tabs.addTab(self.phase_transition_tab, "Phase Transition")

        self.layout.addWidget(self.tabs)

        # 初始化绘图
        self.plot_spins()
        self.plot_evolution()
        self.plot_correlation()
        self.plot_structure_factor()

    def update_model(self):
        try:
            new_L = self.L_spin.value()
            new_T = self.T_spin.value()
            new_algorithm = self.alg_combo.currentText()

            # 检查是否需要重新创建模型
            if new_L != self.L or new_T != self.T or new_algorithm != self.algorithm:
                self.L = new_L
                self.T = new_T
                self.algorithm = new_algorithm

                # 重新创建模型和更新器
                self.model = HeisenbergFCC(L=self.L, T=self.T)
                self.updater = create_updater(self.model, self.algorithm)

                # 重置数据
                self.reset_data()

                QMessageBox.information(self, "Success", "Model updated successfully!")

                # 更新所有视图
                self.plot_spins()
                self.plot_evolution()
                self.plot_correlation()
                self.plot_structure_factor()

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to update model: {str(e)}")

    def reset_data(self):
        self.energy_history = []
        self.magnetization_history = []
        self.steps = []
        self.current_step = 0
        self.plot_evolution()

    def plot_spins(self, clusters: Optional[List[Set[int]]] = None):
        self.spins_canvas.fig.clear()
        ax = self.spins_canvas.fig.add_subplot(111, projection="3d")
        self.vis.plot_spins(self.model, clusters=clusters, ax=ax)
        self.spins_canvas.draw()

    def plot_evolution(self):
        self.evolution_canvas.fig.clear()
        fig = self.evolution_canvas.fig

        # 创建两个子图
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        # 绘制能量演化
        if self.steps:
            ax1.plot(self.steps, self.energy_history, "b-", label="Energy/N")
            ax2.plot(self.steps, self.magnetization_history, "r-", label="|M|")

            ax1.legend()
            ax2.legend()

        ax1.set_xlabel("MC Steps")
        ax1.set_ylabel("Energy per Spin")
        ax1.grid(True)

        ax2.set_xlabel("MC Steps")
        ax2.set_ylabel("Magnetization")
        ax2.grid(True)

        fig.tight_layout()
        self.evolution_canvas.draw()

    def plot_correlation(self):
        self.correlation_canvas.fig.clear()
        correlation = self.model.measure_correlation()
        distance = np.arange(len(correlation))

        ax = self.correlation_canvas.fig.add_subplot(111)
        ax.plot(distance, correlation, "o-")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Correlation")
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_title("Spin-Spin Correlation Function")

        self.correlation_canvas.draw()

    def plot_structure_factor(self):
        self.structure_canvas.fig.clear()
        sf = self.model.measure_structure_factor()
        L = sf.shape[0]

        ax = self.structure_canvas.fig.add_subplot(111)
        im = ax.imshow(
            sf[:, :, L // 2],
            extent=[-np.pi, np.pi, -np.pi, np.pi],
            cmap="hot",
            aspect="auto",
        )
        self.structure_canvas.fig.colorbar(im, ax=ax, label="S(q)")
        ax.set_xlabel("qx")
        ax.set_ylabel("qy")
        ax.set_title("Structure Factor S(q) [qz=0 plane]")

        self.structure_canvas.draw()

    def toggle_animation(self):
        if not self.animating:
            self.animating = True
            self.animate_button.setText("Stop Animation")
            self.timer.start(100)  # 每100毫秒更新一次
        else:
            self.animating = False
            self.animate_button.setText("Start Animation")
            self.timer.stop()

    def update_animation(self):
        # 执行更新
        self.updater.update()
        self.current_step += 1

        # 获取簇信息
        clusters = None
        if hasattr(self.updater, "cluster"):
            clusters = [self.updater.cluster]
        elif hasattr(self.updater, "clusters"):
            clusters = self.updater.clusters

        # 每隔一定步数记录物理量
        if self.current_step % self.measurement_interval == 0:
            energy = self.model.energy / self.model.N
            magnetization = np.linalg.norm(self.model.calculate_magnetization())

            self.energy_history.append(energy)
            self.magnetization_history.append(magnetization)
            self.steps.append(self.current_step)

            # 更新图表
            self.plot_evolution()

            # 定期更新关联函数和结构因子
            if self.current_step % (self.measurement_interval * 10) == 0:
                self.plot_correlation()
                self.plot_structure_factor()

        # 更新自旋构型显示
        self.plot_spins(clusters=clusters)

    def run_phase_transition_analysis(self):
        """执行相变分析"""
        try:
            # 获取参数
            T_start = self.T_start_spin.value()
            T_end = self.T_end_spin.value()
            cooling_steps = self.cooling_steps_spin.value()

            # 创建模拟实例
            sim = AnnealingSimulation(
                L=self.L,
                T_start=T_start,
                T_end=T_end,
                cooling_steps=cooling_steps,
                mc_steps_per_T=1000,  # 可以添加控制
                thermalization_steps=100,  # 可以添加控制
            )

            # 运行模拟
            results = sim.run()

            # 显示结果
            self.plot_phase_transition_results(results)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to run analysis: {str(e)}")

    def plot_phase_transition_results(self, results: Dict):
        """绘制相变分析结果"""
        self.phase_canvas.fig.clear()

        # 创建2x2的子图
        ((ax1, ax2), (ax3, ax4)) = self.phase_canvas.fig.subplots(2, 2)

        T = results["temperature"]

        # 能量
        ax1.plot(T, results["energy"], "o-")
        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy per spin")
        ax1.grid(True)

        # 磁化强度
        ax2.plot(T, results["magnetization"], "o-")
        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Magnetization")
        ax2.grid(True)

        # 比热容
        ax3.plot(T, results["specific_heat"], "o-")
        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Specific Heat")
        ax3.grid(True)

        # 磁化率
        ax4.plot(T, results["susceptibility"], "o-")
        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.grid(True)

        self.phase_canvas.fig.tight_layout()
        self.phase_canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractiveApp()
    window.show()
    sys.exit(app.exec_())
