"""
Wolff单群集更新算法实现
"""

import numpy as np
from typing import Set, Tuple
from HeisenbergFCC import HeisenbergFCC


class WolffUpdate:
    def __init__(self, model: HeisenbergFCC):
        self.model = model
        self.beta = model.beta
        self.J = model.J
        self.cluster: Set[int] = set()
        self.projection_dir = None

    def _generate_projection_direction(self):
        dir = np.random.randn(3)
        self.projection_dir = dir / np.linalg.norm(dir)

    def _build_cluster(self, start_site: int):
        stack = [start_site]
        self.cluster = {start_site}
        spins = self.model.spins
        r = self.projection_dir

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

    def update(self):
        self._generate_projection_direction()
        start_site = np.random.randint(0, self.model.N)
        self._build_cluster(start_site)

        # 反射群集中的所有自旋
        r = self.projection_dir
        for i in self.cluster:
            Si_proj = np.dot(self.model.spins[i], r)
            self.model.spins[i] -= 2 * Si_proj * r

        # 更新系统能量
        self.model.energy = self.model.calculate_total_energy()


if __name__ == "__main__":
    model = HeisenbergFCC(L=4, T=2.0)
    updater = WolffUpdate(model)

    # 测试更新
    print(f"Initial energy: {model.energy}")
    for _ in range(10):
        updater.update()
        print(f"Cluster size: {len(updater.cluster)}, Energy: {model.energy}")
