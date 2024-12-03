"""
模拟控制类，支持单系统和并行模拟
"""

import multiprocessing as mp
import numpy as np
from typing import List, Dict
from HeisenbergFCC import HeisenbergFCC
from PhysicalQuantities import PhysicalQuantities
from updaters import create_updater
import matplotlib.pyplot as plt
from FSAnalysis import FSAnalysis
from tqdm import tqdm


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


if __name__ == "__main__":
    # 测试并行模拟
    L_values = [4, 8]
    T_values = [1.0, 2.0]
    updater_types = ["single", "sw", "wolff"]

    sim = ParallelSimulation(L_values, T_values, updater_types)
    sim.run()

    # 获取能量数据并打印
    energies = sim.get_observable("E")
    for key, values in energies.items():
        print(f"L={key[0]}, T={key[1]}, updater={key[2]}")
        print(f"Mean energy: {np.mean(values)}")
