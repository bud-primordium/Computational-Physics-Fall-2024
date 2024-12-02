"""
Swendsen-Wang群集更新算法的实现
基于将海森堡自旋投影到随机方向上的原理
"""

import numpy as np
from typing import List, Set
from HeisenbergFCC import HeisenbergFCC
from Visualization import Visualization


class SwendsenWangUpdate:
    """Swendsen-Wang群集更新"""

    def __init__(self, model: HeisenbergFCC):
        """
        初始化更新器

        参数：
            model (HeisenbergFCC): FCC海森堡模型实例
        """
        self.model = model
        self.beta = model.beta
        self.J = model.J
        self.N = model.N
        self.neighbors = model.neighbors

        # 用于存储群集标签
        self.labels = np.arange(self.N)
        # 存储群集列表，供可视化使用
        self.clusters: List[Set[int]] = []
        # 随机投影方向
        self.projection_dir = None

    def _reset_labels(self):
        """重置群集标签"""
        self.labels = np.arange(self.N)

    def _find(self, x):
        """并查集的查找操作"""
        if self.labels[x] != x:
            self.labels[x] = self._find(self.labels[x])
        return self.labels[x]

    def _union(self, x, y):
        """并查集的合并操作"""
        x_root = self._find(x)
        y_root = self._find(y)
        if x_root != y_root:
            self.labels[y_root] = x_root

    def _generate_projection_direction(self):
        """生成随机投影方向"""
        # 在单位球面上均匀采样
        dir = np.random.randn(3)
        dir /= np.linalg.norm(dir)
        self.projection_dir = dir

    def _build_clusters(self):
        """构建群集"""
        self._reset_labels()
        self._generate_projection_direction()

        spins = self.model.spins
        beta_J = self.beta * self.J
        r = self.projection_dir

        for i in range(self.N):
            # 计算自旋i在投影方向上的分量
            Si_proj = np.dot(spins[i], r)

            for j in self.neighbors[i]:
                if i < j:  # 避免重复处理
                    # 计算自旋j在投影方向上的分量
                    Sj_proj = np.dot(spins[j], r)

                    # 只有当投影自旋指向相同方向时才可能连接
                    if Si_proj * Sj_proj > 0:
                        # 计算连接概率
                        pij = 1 - np.exp(-2 * beta_J * Si_proj * Sj_proj)
                        if np.random.rand() < pij:
                            self._union(i, j)

    def _extract_clusters(self):
        """提取群集列表"""
        # 确保所有节点的根节点已更新
        for i in range(self.N):
            self._find(i)

        clusters_dict = {}
        for i in range(self.N):
            root = self.labels[i]
            if root not in clusters_dict:
                clusters_dict[root] = set()
            clusters_dict[root].add(i)

        # 将群集转换为列表
        self.clusters = list(clusters_dict.values())

    def _update_clusters(self):
        """更新群集内的自旋"""
        spins = self.model.spins
        r = self.projection_dir

        for cluster in self.clusters:
            if np.random.rand() < 0.5:  # 以概率1/2翻转群集
                cluster_indices = list(cluster)
                # 对群集内所有自旋关于投影方向进行反射
                for i in cluster_indices:
                    Si_proj = np.dot(spins[i], r)
                    spins[i] = spins[i] - 2 * Si_proj * r

        # 更新模型中的自旋构型
        self.model.spins = spins
        # 更新模型的总能量
        self.model.energy = self.model.calculate_total_energy()

    def update(self):
        """执行一次Swendsen-Wang群集更新"""
        self._build_clusters()
        self._extract_clusters()
        self._update_clusters()


if __name__ == "__main__":
    # 测试代码
    L = 4
    T = 0.1  # 温度
    model = HeisenbergFCC(L=L, T=T)
    updater = SwendsenWangUpdate(model)

    # 进行一次更新
    updater.update()

    # 可视化群集
    vis = Visualization()
    vis.plot_clusters(model, updater.clusters)
