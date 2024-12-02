"""
3D FCC海森堡模型的核心实现
包含晶格初始化、邻居表构建、能量计算和物理量测量
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.optimize import curve_fit


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
