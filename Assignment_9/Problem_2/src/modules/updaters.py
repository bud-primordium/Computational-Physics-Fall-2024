"""
单自旋Metropolis、Swendsen-Wang和Wolff更新算法的统一实现
"""

import numpy as np
from typing import List, Set, Dict, Optional
from HeisenbergFCC import HeisenbergFCC


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
