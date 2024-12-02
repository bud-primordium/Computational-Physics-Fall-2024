import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    fss = FSAnalysis()
    # Test code here...
