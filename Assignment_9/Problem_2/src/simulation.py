"""
模拟控制类，支持单系统和并行模拟
"""

import multiprocessing as mp
import numpy as np
from typing import List, Dict
from HeisenbergFCC import HeisenbergFCC
from PhysicalQuantities import PhysicalQuantities
from updaters import create_updater


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
