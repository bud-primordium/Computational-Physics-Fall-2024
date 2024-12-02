"""
Metropolis单自旋翻转更新算法的实现
"""

import numpy as np
from typing import Optional
from HeisenbergFCC import HeisenbergFCC


class SingleSpinUpdate:
    """单自旋Metropolis更新"""

    def __init__(self, model: HeisenbergFCC, delta_theta_max: float = 0.1):
        """
        初始化更新器

        参数：
            model (HeisenbergFCC): FCC海森堡模型实例
            delta_theta_max (float): 最大角度变化（弧度），默认为0.1
        """
        self.model = model
        # 确保初始步长不超过π
        self.delta_theta_max = min(delta_theta_max, np.pi)

        # 统计接受率
        self.n_proposed = 0
        self.n_accepted = 0

    def _random_rotation(self, spin: np.ndarray) -> np.ndarray:
        """
        生成试探自旋，通过在原自旋附近添加小的随机扰动得到

        参数：
            spin (np.ndarray): 当前自旋向量

        返回：
            new_spin (np.ndarray): 新的归一化自旋向量
        """
        # 随机生成一个小的旋转角度
        delta_theta = np.random.uniform(0, self.delta_theta_max)
        # 随机生成一个旋转轴（单位向量）
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)
        # 使用Rodrigues公式计算旋转
        new_spin = (
            spin * np.cos(delta_theta)
            + np.cross(axis, spin) * np.sin(delta_theta)
            + axis * np.dot(axis, spin) * (1 - np.cos(delta_theta))
        )
        return new_spin / np.linalg.norm(new_spin)

    def update(self) -> None:
        """执行一次单自旋更新"""
        # 随机选择一个自旋索引
        i = np.random.randint(0, self.model.N)
        old_spin = self.model.get_spin(i).copy()

        # 计算当前自旋的能量
        old_energy = self.model.calculate_site_energy(i)

        # 生成新的自旋方向
        new_spin = self._random_rotation(old_spin)

        # 临时更新自旋以计算新能量
        self.model.spins[i] = new_spin
        new_energy = self.model.calculate_site_energy(i)

        # 计算能量差
        delta_E = new_energy - old_energy

        # Metropolis判据
        self.n_proposed += 1
        if delta_E <= 0 or np.random.rand() < np.exp(-self.model.beta * delta_E):
            # 接受新构型，更新系统总能量
            self.n_accepted += 1
            self.model.energy += delta_E
        else:
            # 拒绝新构型，恢复旧的自旋
            self.model.spins[i] = old_spin

    def sweep(self) -> None:
        """对所有自旋进行一次完整的更新（一个蒙特卡洛步）"""
        for _ in range(self.model.N):
            self.update()

    @property
    def acceptance_rate(self) -> float:
        """计算接受率"""
        if self.n_proposed == 0:
            return 0.0
        return self.n_accepted / self.n_proposed

    def reset_counters(self) -> None:
        """重置计数器"""
        self.n_proposed = 0
        self.n_accepted = 0

    def adjust_step_size(
        self, target_rate: float = 0.5, tolerance: float = 0.05
    ) -> None:
        """
        调整最大角度变化以达到目标接受率

        参数：
            target_rate (float): 目标接受率，默认为0.5
            tolerance (float): 可接受的偏差，默认为0.05
        """
        current_rate = self.acceptance_rate
        if self.n_proposed == 0:
            return  # 尚未进行任何更新，不调整步长
        if abs(current_rate - target_rate) > tolerance:
            # 根据接受率调整步长
            adjustment_factor = current_rate / target_rate
            self.delta_theta_max *= adjustment_factor
            # 限制步长在合理范围内
            self.delta_theta_max = min(max(self.delta_theta_max, 1e-5), np.pi)
        # 在调整步长后重置计数器
        self.reset_counters()


if __name__ == "__main__":
    # 创建模型实例
    L = 4
    T = 3  # 温度
    model = HeisenbergFCC(L=L, T=T)
    updater = SingleSpinUpdate(model)

    # 进行蒙特卡洛模拟
    num_sweeps = 1000
    for sweep in range(num_sweeps):
        updater.sweep()
        # 每隔一定步数调整步长
        if (sweep + 1) % 100 == 0:
            # 记录当前接受率
            acceptance_rate = updater.acceptance_rate
            # 调整步长
            updater.adjust_step_size()
            # 打印信息
            print(
                f"Sweep {sweep + 1}, Acceptance Rate: {acceptance_rate:.3f}, "
                f"Delta Theta Max: {updater.delta_theta_max:.5f}"
            )
