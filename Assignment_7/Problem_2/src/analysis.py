"""径向薛定谔方程求解器的分析模块

包含波函数处理、能量分析和收敛性研究等功能。
对求解得到的结果进行验证、分析和后处理。

Classes:
   WavefunctionProcessor: 波函数处理器
   EnergyAnalyzer: 能量分析器
   ConvergenceAnalyzer: 收敛性分析器
"""

import numpy as np
from typing import Dict, Tuple
import logging
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)


class WavefunctionProcessor:
    """波函数处理类(非均匀网格版本)"""

    def __init__(self, r: np.ndarray, l: int, delta: float):
        """初始化

        Parameters
        ----------
        r : np.ndarray
            非均匀径向网格点
        l : int
            角量子数
        delta : float
            网格变换参数
        """
        self.r = r
        self.l = l
        self.delta = delta
        self.j = np.arange(len(r))  # 均匀的j网格
        self.r_p = r[-1] / (np.exp(delta * (len(r) - 1)) - 1)

        # 计算网格变换的导数
        self.dr_dj = self.r_p * delta * np.exp(delta * self.j)

    def get_derivatives(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算波函数在非均匀网格上的导数

        Parameters
        ----------
        u : np.ndarray
            波函数值

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            一阶和二阶导数
        """
        du_dr = np.gradient(u, self.r)
        d2u_dr2 = np.gradient(du_dr, self.r)
        return du_dr, d2u_dr2

    def analyze_asymptotic(
        self, u: np.ndarray, num_points: int = 10
    ) -> Tuple[float, float]:
        """分析r→0时波函数的渐进行为
        Returns:
            m: r→0时的幂次（应接近l）
            n_analyze: exp(-r/n)中的n
        """
        # 选取近原点的几个点
        j_near_zero = self.j[:num_points]
        r_near_zero = self.r[:num_points]
        u_near_zero = u[:num_points]

        # 对数拟合
        mask = (r_near_zero > 0) & (np.abs(u_near_zero) > 1e-15)
        if not np.any(mask):
            return self.l, 1.0  # 默认值改为l

        x = np.log(r_near_zero[mask])
        y = np.log(np.abs(u_near_zero[mask]))

        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]

        # 计算n_analyze (从c中获取)
        n_analyze = np.exp(c)

        return m, n_analyze

    def get_r0_values(self, u: np.ndarray) -> Tuple[float, float]:
        """获取r=0处的函数值和导数"""
        # 可能弃用，因为后面绘制的是r=r_min开始的
        if self.l == 0:
            # l=0时,外推u(0),du/dr(0)=0
            du_dr, _ = self.get_derivatives(u)
            u0 = u[0] - self.r[0] * du_dr[0]  # 一阶泰勒展开
            return u0, 0.0
        else:
            # l>0时,u(0)=0,通过渐进行为确定du/dr(0)
            coef, power = self.analyze_asymptotic(u)
            if abs(power - (self.l + 1)) > 0.5:
                logger.warning(f"渐进指数{power:.2f}与预期值{self.l+1}差异较大")
            return 0.0, coef * (self.l + 1)

    def normalize_wavefunction(
        self, u: np.ndarray, tol: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """归一化波函数"""
        # 数值稳定性检查
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            logger.error("波函数包含无效值")
            raise ValueError("Invalid wavefunction values")

        # 处理r=r_min的情况
        # u0, _ = self.get_r0_values(u)

        # 拟合的函数模型：r^(l+1)
        def r_behavior(r, a):
            return a * r ** (self.l + 1)

        # 使用前几个点拟合
        fit_indices = slice(1, 7)  # 使用 u 第2~7个点（索引从1到6）
        r_fit = self.r[fit_indices]
        u_fit = u[fit_indices]

        # 执行拟合
        popt, _ = curve_fit(r_behavior, r_fit, u_fit)

        # 根据拟合结果外推 u_full[0]
        u_full = np.copy(u)
        u_full[0] = r_behavior(self.r[0], *popt)

        # 计算归一化常数(考虑非均匀网格的积分权重)
        mask = np.abs(u_full) > 1e-15
        if not np.any(mask):
            logger.error("波函数全为零")
            raise ValueError("Zero wavefunction")

        norm = np.sqrt(np.trapz(u_full[mask] * u_full[mask], self.r[mask]))
        if norm < 1e-15:
            logger.error("归一化常数接近零")
            raise ValueError("Normalization constant too small")

        u_norm = u_full / norm

        # 计算R(r) = u(r)/r
        R = np.zeros_like(u_norm)
        nonzero_r = self.r > 1e-10
        R[nonzero_r] = u_norm[nonzero_r] / self.r[nonzero_r]
        # 补丁，l>0时r=r_min处的处理
        # 使用前几个点拟合
        fit_indices = slice(1, 7)  # 使用 R 第2~7个点（索引从1到6）
        r_fit = self.r[fit_indices]
        R_fit = R[fit_indices]

        # 执行拟合
        popt, _ = curve_fit(r_behavior, r_fit, R_fit)

        # 根据拟合结果外推 R_full[0]
        R_full = np.copy(R)
        R_full[0] = r_behavior(self.r[0], *popt)
        # # r=0处的处理
        # if self.l == 0:
        #     # 使用洛必达法则: lim(r→0) u(r)/r = du/dr(0)
        #     R[0] = du0_dr / norm
        # else:
        #     # 使用渐进行为
        #     coef, _ = self.analyze_asymptotic(u_norm)
        #     R[0] = coef * self.l

        self._verify_normalization(R, tol)
        return u_norm, R

    def _verify_normalization(self, R: np.ndarray, tol: float):
        """验证波函数归一化(考虑非均匀网格)"""
        mask = np.isfinite(R) & (np.abs(R) < 1e10)
        if not np.any(mask):
            logger.error("无有效值用于归一化检验")
            raise ValueError("No valid values for normalization check")

        # 在非均匀网格上积分
        integrand = R[mask] * R[mask] * self.r[mask] * self.r[mask]
        norm = np.trapz(integrand, self.r[mask])

        if not np.isfinite(norm):
            logger.error("归一化积分结果无效")
            raise ValueError("Invalid normalization integral")

        if not (1 - tol < norm < 1 + tol):
            logger.error(f"归一化检验失败: ∫|R(r)|²r²dr = {norm:.6f}")
            raise ValueError(f"Normalization check failed: {norm:.6f}")


class EnergyAnalyzer:
    """能量分析类"""

    def __init__(self, theoretical_values: Dict):
        """初始化

        Parameters
        ----------
        theoretical_values : Dict
            理论值字典
        """
        self.theoretical_values = theoretical_values

    def compare_with_theory(self, E: float, V_type: str, n: int, l: int) -> dict:
        """与理论值比较

        Parameters
        ----------
        E : float
            计算得到的能量
        V_type : str
            势能类型
        n : int
            主量子数
        l : int
            角量子数

        Returns
        -------
        dict
            比较结果字典
        """
        state = (n, l)
        theory = self.theoretical_values[V_type].get(state)

        result = {
            "numerical_E": E,
            "theoretical_E": theory,
            "relative_error": None,
            "status": "unknown",
        }

        if theory is not None:
            rel_error = abs((E - theory) / theory) * 100
            result.update(
                {
                    "relative_error": rel_error,
                    "status": "good" if rel_error < 1.0 else "warning",
                }
            )

        return result


class ConvergenceAnalyzer:
    """收敛性分析类"""

    def __init__(self, energy_analyzer: EnergyAnalyzer):
        """初始化

        Parameters
        ----------
        energy_analyzer : EnergyAnalyzer
            能量分析器实例
        """
        self.energy_analyzer = energy_analyzer

    def analyze_grid_convergence(self, solver, n_values: list) -> dict:
        """分析不同网格点数的收敛性

        Parameters
        ----------
        solver : object
            求解器实例
        n_values : list
            网格点数列表

        Returns
        -------
        dict
            收敛性分析结果
        """
        results = {"n_points": n_values, "energies": [], "errors": [], "delta_h": []}

        for n in n_values:
            try:
                # 使用新的网格点数求解
                E = solver.solve_with_points(n)

                # 计算误差
                analysis = self.energy_analyzer.compare_with_theory(
                    E, solver.config.V_type, solver.config.n, solver.config.l
                )

                results["energies"].append(E)
                if analysis["relative_error"] is not None:
                    results["errors"].append(analysis["relative_error"])
                results["delta_h"].append(6 / n)  # r=rp(e^t-1)+r_min t_max = 6

            except Exception as e:
                logger.warning(f"网格点数{n}的计算失败: {str(e)}")
                continue

        return results
