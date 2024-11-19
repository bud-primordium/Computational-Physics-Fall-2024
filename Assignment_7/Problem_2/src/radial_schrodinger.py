"""径向薛定谔方程求解器

用于求解两种势能:
1. 氢原子: V(r) = -1/r
2. 锂原子: V(r) = -Z_ion/r * erf(r/sqrt(2)/r_loc) + exp(-1/2*(r/r_loc)^2)*[...]

支持两种求解方法:
1. 打靶法(shooting): 从外向内积分,寻找满足边界条件的能量本征值
2. 有限差分法(finite difference): 构建矩阵直接求解本征值问题

结果验证:
1. 解析解对比 (氢原子)
2. 理论值对照
3. 收敛性分析
"""

# Part 1 Basic Setup
import numpy as np
from scipy.sparse import diags, linalg
from scipy.special import erf
from scipy.optimize import root_scalar
from scipy.sparse import spmatrix
from scipy.special import factorial
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import logging
import warnings
import copy

# 本题取h_bar=m=1

# 忽略一些数值计算的警告
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class SolverConfig:
    """求解器配置类

    Attributes:
        r_max: 最大径向距离
        j_max: 径向网格点数
        delta: 网格变换参数
        l: 角量子数
        n: 主量子数
        n_states: 需要求解的本征态数量
        V_type: 势能类型('hydrogen'或'lithium')
        method: 求解方法('shooting'或'fd')
        tol: 收敛精度
    """

    r_max: float = 30.0  # 最大半径 (Bohr)
    j_max: int = 1000  # 网格点数
    delta: float = 0.006  # 网格间距
    l: int = 0  # 角量子数
    n: int = 1  # 主量子数
    n_states: int = 3  # 求解本征态数量
    V_type: str = "hydrogen"  # 势能类型
    method: str = "shooting"  # 求解方法
    tol: float = 1e-6  # 收敛精度


def get_theoretical_values() -> Dict:
    """理论参考值(Hartree单位)

    Returns:
        Dict: 不同原子的能量本征值字典
            格式: {原子类型: {(n,l): 能量}}
    """
    return {
        "hydrogen": {
            # (n,l): E
            (1, 0): -0.5,  # 1s
            (2, 0): -0.125,  # 2s
            (2, 1): -0.125,  # 2p
            (3, 0): -1 / 18,  # 3s
            (3, 1): -1 / 18,  # 3p
            (3, 2): -1 / 18,  # 3d
        },
        "lithium": {
            # 原本参考值来自学长 www.github.com/ShangkunLi/Computational_Physics/ 但感觉有问题，后来随便设定了一个与Z_ion大致自洽的
            (1, 0): -0.2,  # 1s
            (2, 0): -0.05,  # 2s
            (2, 1): -0.05,  # 2p
        },
    }


# 设置logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Part 2 Tools
# 网格和势能相关的基础工具函数
class RadialGrid:
    """径向网格类，处理非均匀网格的生成和变换"""

    def __init__(self, config: SolverConfig):
        """初始化网格

        Args:
            config: 求解器配置对象
        """
        self.config = config
        self.setup_grid()

    def setup_grid(self):
        """设置非均匀网格
        使用变换 r = r_p[exp(j*delta) - 1]
        在原子核附近网格密集,远处稀疏
        """
        self.j = np.arange(self.config.j_max + 1)
        # 计算变换参数
        self.r_p = self.config.r_max / (
            np.exp(self.config.delta * self.config.j_max) - 1
        )
        # 生成径向网格点
        self.r = self.r_p * (np.exp(self.config.delta * self.j) - 1)
        # 计算变换后的网格间距
        self.dr = np.diff(self.r)

    def get_grid_info(self) -> dict:
        """返回网格信息"""
        return {
            "r": self.r,
            "dr": self.dr,
            "r_min": self.r[0],
            "r_max": self.r[-1],
            "n_points": len(self.r),
        }


class PotentialFunction:
    """势能函数类"""

    @staticmethod
    def V_hydrogen(r: np.ndarray) -> np.ndarray:
        """氢原子势能
        V(r) = -1/r (原子单位)

        Args:
            r: 径向距离数组

        Returns:
            势能数组
        """
        return -1 / r

    @staticmethod
    def V_lithium(
        r: np.ndarray,
        Z_ion: float = 3,
        r_loc: float = 0.4,
        C1: float = -14.0093922,
        C2: float = 9.5099073,
        C3: float = -1.7532723,
        C4: float = 0.0834586,
    ) -> np.ndarray:
        """锂原子势能
        包含库仑势和局域赝势项

        Args:
            r: 径向距离数组
            Z_ion: 有效核电荷
            r_loc: 局域化参数
            C1-C4: 势能参数

        Returns:
            势能数组
        """
        # 库仑项
        term1 = -Z_ion / r * erf(r / (np.sqrt(2) * r_loc))
        # 局域项
        exp_term = np.exp(-0.5 * (r / r_loc) ** 2)
        term2 = exp_term * (
            C1 + C2 * (r / r_loc) ** 2 + C3 * (r / r_loc) ** 4 + C4 * (r / r_loc) ** 6
        )
        return term1 + term2

    @classmethod
    def get_potential(cls, V_type: str):
        """获取势能函数

        Args:
            V_type: 势能类型('hydrogen'或'lithium')

        Returns:
            对应的势能函数
        """
        if V_type == "hydrogen":
            return cls.V_hydrogen
        elif V_type == "lithium":
            return cls.V_lithium
        else:
            raise ValueError(f"未知的势能类型: {V_type}")


class WavefunctionTools:
    """波函数工具类"""

    @staticmethod
    def get_analytic_hydrogen(r: np.ndarray, n: int, l: int) -> Optional[np.ndarray]:
        """获取氢原子解析解
        Ref:https://quantummechanics.ucsd.edu/ph130a/130_notes/node233.html

        Args:
            r: 径向距离数组
            n: 主量子数
            l: 角量子数

        Returns:
            解析波函数数组，如果无解析解则返回None
        """

        if n == 1 and l == 0:  # 1s
            return 2 * np.exp(-r)
        elif n == 2 and l == 0:  # 2s
            return np.sqrt(2) / 4 * (2 - r) * np.exp(-r / 2)
        elif n == 2 and l == 1:  # 2p
            return np.sqrt(2) / (4 * np.sqrt(3)) * r * np.exp(-r / 2)
        elif n == 3 and l == 0:  # 3s
            return 2 / 243 * np.sqrt(3) * (27 - 18 * r + 2 * r**2) * np.exp(-r / 3)
        elif n == 3 and l == 1:  # 3p
            return 2 / 81 * np.sqrt(6) * r * (6 - r) * np.exp(-r / 3)
        return None

    @staticmethod
    def count_nodes(u: np.ndarray, eps: float = 1e-10) -> int:
        """计算波函数节点数

        Args:
            u: 波函数数组
            eps: 判定零点的阈值

        Returns:
            节点数
        """
        # 忽略小波动
        u_filtered = np.where(abs(u) < eps, 0, u)
        # 寻找符号变化点
        return len(np.where(np.diff(np.signbit(u_filtered)))[0])

    @staticmethod
    def verify_nodes(u: np.ndarray, n: int, l: int) -> bool:
        """验证节点数是否符合量子力学要求
        n-l-1 = 节点数

        Args:
            u: 波函数数组
            n: 主量子数
            l: 角量子数

        Returns:
            是否符合要求
        """
        nodes = WavefunctionTools.count_nodes(u)
        expected_nodes = n - l - 1
        return nodes == expected_nodes


def get_initial_energy_range(config: SolverConfig) -> Tuple[float, float]:
    """获取能量搜索范围

    Args:
        config: 求解器配置

    Returns:
        (E_min, E_max): 能量搜索范围
    """
    if config.V_type == "hydrogen":
        # 氢原子能量: E_n = -1/(2n^2)
        E_theo = -1.0 / (2 * config.n**2)
        # 使用更大的搜索范围
        E_min = E_theo * 1.5  # 扩大搜索范围下限
        E_max = E_theo * 0.5  # 扩大搜索范围上限
    else:
        # 锂原子能量估计
        E_theo = -0.1 / config.n**2
        E_min = E_theo * 1.5
        E_max = E_theo * 0.5

    return E_min, E_max


# Part 3 Wavefunction
class WavefunctionProcessor:
    """波函数处理类"""

    def __init__(self, r: np.ndarray, l: int):
        """初始化

        Args:
            r: 径向网格点
            l: 角量子数
        """
        self.r = r
        self.l = l

    def normalize_wavefunction(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """归一化波函数

        Args:
            u: 原始波函数u(r)

        Returns:
            (u_norm, R): 归一化的u(r)和R(r)
        """
        # 添加数值稳定性检查
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            logger.warning("波函数包含无效值")
            return u, u / self.r

        # 处理可能的零值
        mask = np.abs(u) > 1e-15
        if not np.any(mask):
            logger.warning("波函数全为零")
            return u, np.zeros_like(u)

        # 计算归一化常数，仅使用有效值
        norm_u = np.sqrt(np.trapz(u[mask] * u[mask], self.r[mask]))

        # 检查归一化常数
        if norm_u < 1e-15:
            logger.warning("归一化常数接近零")
            return u, np.zeros_like(u)

        u_norm = u / norm_u

        # 计算R(r) = u(r)/r，注意r=0处的处理
        R = np.zeros_like(u_norm)
        nonzero_r = self.r > 1e-10
        R[nonzero_r] = u_norm[nonzero_r] / self.r[nonzero_r]

        # r=0处的极限值处理
        if self.l == 0:
            # l=0时，使用临近点的值估计R(0)
            if len(R[nonzero_r]) > 0:
                R[0] = R[nonzero_r][0]
        else:
            # l>0时，R(0)=0
            R[0] = 0

        # 验证归一化
        self._verify_normalization(R)

        return u_norm, R

    def _verify_normalization(self, R: np.ndarray, tol: float = 1e-3):
        """验证波函数归一化"""
        # 仅使用有效值进行归一化检查
        mask = np.isfinite(R) & (np.abs(R) < 1e10)
        if not np.any(mask):
            logger.warning("无有效值用于归一化检查")
            return

        norm = np.trapz(R[mask] * R[mask] * self.r[mask] * self.r[mask], self.r[mask])

        if not np.isfinite(norm):
            logger.warning("归一化积分结果无效")
        elif not (1 - tol < norm < 1 + tol):
            logger.warning(f"归一化可能有问题: ∫|R(r)|²r²dr = {norm:.6f}")

    def analyze_wavefunction(
        self, u: np.ndarray, R: np.ndarray, R_analytic: Optional[np.ndarray] = None
    ) -> dict:
        """分析波函数性质

        Args:
            u: 归一化的u(r)
            R: 归一化的R(r)
            R_analytic: 解析解(如果有)

        Returns:
            分析结果字典
        """
        analysis = {
            "nodes": WavefunctionTools.count_nodes(u),
            "max_amplitude": np.max(np.abs(R)),
            "r_max_prob": self.r[np.argmax(R * R * self.r * self.r)],
            "norm_check": np.trapz(R * R * self.r * self.r, self.r),
        }

        # 如果有解析解，计算误差
        if R_analytic is not None:
            rel_error = np.abs(R - R_analytic) / np.max(np.abs(R_analytic))
            analysis.update(
                {
                    "max_rel_error": np.max(rel_error),
                    "avg_rel_error": np.mean(rel_error),
                    "error_distribution": rel_error,
                }
            )

        return analysis


class EnergyAnalyzer:
    """能量分析类"""

    def __init__(self, theoretical_values: Dict):
        """初始化

        Args:
            theoretical_values: 理论值字典
        """
        self.theoretical_values = theoretical_values

    def compare_with_theory(self, E: float, V_type: str, n: int, l: int) -> dict:
        """与理论值比较

        Args:
            E: 计算得到的能量
            V_type: 势能类型
            n: 主量子数
            l: 角量子数

        Returns:
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

    def print_energy_analysis(self, analysis: dict):
        """打印能量分析结果

        Args:
            analysis: compare_with_theory返回的字典
        """
        print("\n能量分析:")
        print(f"数值结果: E = {analysis['numerical_E']:.6f} Hartree")

        if analysis["theoretical_E"] is not None:
            print(f"理论数值: E = {analysis['theoretical_E']:.6f} Hartree")
            print(f"相对误差: {analysis['relative_error']:.6f}%")

            if analysis["status"] == "warning":
                print("警告: 相对误差超过1%!")


class ConvergenceAnalyzer:
    """收敛性分析类"""

    def __init__(self, energy_analyzer: EnergyAnalyzer):
        """初始化

        Args:
            energy_analyzer: 能量分析器实例
        """
        self.energy_analyzer = energy_analyzer

    def analyze_grid_convergence(self, solver, n_values: List[int]) -> dict:
        """分析不同网格点数的收敛性

        Args:
            solver: 求解器实例
            n_values: 网格点数列表

        Returns:
            收敛性分析结果
        """
        results = {"n_points": n_values, "energies": [], "errors": [], "delta_h": []}

        for n in n_values:
            # 使用新的网格点数求解
            E = solver.solve_with_points(n)

            # 计算误差
            analysis = self.energy_analyzer.compare_with_theory(
                E, solver.config.V_type, solver.config.n, solver.config.l
            )

            results["energies"].append(E)
            if analysis["relative_error"] is not None:
                results["errors"].append(analysis["relative_error"])
            results["delta_h"].append(solver.config.r_max / n)

        return results


# Part 4 ShootingSolver
class ShootingSolver:
    """打靶法求解器"""

    def __init__(self, grid: RadialGrid, V: callable, l: int):
        """初始化打靶法求解器

        Args:
            grid: 径向网格对象
            V: 势能函数
            l: 角量子数
        """
        self.grid = grid
        self.V = V
        self.l = l
        self.delta = grid.config.delta

    def integrate_inward(self, E: float) -> np.ndarray:
        """从外向内积分

        使用RK4方法求解变换后的方程:
        d²v/dj² - (δ²/4)v = r_p²δ²e^(2jδ)[V(r) - E - l(l+1)/(2r²)]v

        Args:
            E: 能量

        Returns:
            波函数u(r)
        """
        # 初始化波函数，使用高精度数据类型
        v = np.zeros(self.grid.config.j_max + 1, dtype=np.float64)
        dvdj = np.zeros(self.grid.config.j_max + 1, dtype=np.float64)

        # 优化边界条件
        v[-1] = 0.0
        v[-2] = 1e-12  # 使用更小的初始值以提高数值稳定性

        # 辅助函数：计算有效势能
        def get_effective_potential(r: float) -> float:
            """计算包含向心势的有效势能"""
            # 防止r=0时的除零错误
            if r < 1e-10:
                return -E + 1e10  # 在原点附近使用大的排斥势
            return self.V(r) - E - self.l * (self.l + 1) / (2 * r * r)

        # 辅助函数：计算v的二阶导数
        def d2vdj2(j: float, v_val: float) -> float:
            """计算波函数的二阶导数"""
            r = self.grid.r_p * (np.exp(self.delta * j) - 1)
            exp_term = np.exp(2 * self.delta * j)
            # 数值稳定性：限制指数项的增长
            if exp_term > 1e30:
                exp_term = 1e30

            V_eff = get_effective_potential(r)

            return (
                (self.delta**2 / 4) * v_val
                - self.grid.r_p**2 * self.delta**2 * exp_term * V_eff * v_val
            )

        # 初始步长
        h = -self.delta  # 负步长，向内积分
        base_step = h

        # RK4向内积分
        for j in range(self.grid.config.j_max - 1, -1, -1):
            r = self.grid.r[j]

            # 自适应步长控制
            if r < 1.0:  # 在靠近原子核区域使用更小的步长
                h = base_step * min(1.0, r)

            try:
                # RK4步骤
                k1v = h * dvdj[j + 1]
                k1dv = h * d2vdj2(j + 1, v[j + 1])

                k2v = h * (dvdj[j + 1] + 0.5 * k1dv)
                k2dv = h * d2vdj2(j + 0.5, v[j + 1] + 0.5 * k1v)

                k3v = h * (dvdj[j + 1] + 0.5 * k2dv)
                k3dv = h * d2vdj2(j + 0.5, v[j + 1] + 0.5 * k2v)

                k4v = h * (dvdj[j + 1] + k3dv)
                k4dv = h * d2vdj2(j, v[j + 1] + k3v)

                # 更新波函数和导数
                v_new = v[j + 1] + (k1v + 2 * k2v + 2 * k3v + k4v) / 6
                dvdj_new = dvdj[j + 1] + (k1dv + 2 * k2dv + 2 * k3dv + k4dv) / 6

                # 数值稳定性检查
                if np.isnan(v_new) or np.abs(v_new) > 1e15:
                    logger.warning(f"数值不稳定 at j={j}, r={r:.6e}, 尝试重置值")
                    # 如果值变得不稳定，使用前一个稳定值
                    v_new = v[j + 1] * 0.9
                    dvdj_new = dvdj[j + 1] * 0.9

                v[j] = v_new
                dvdj[j] = dvdj_new

            except Exception as e:
                logger.error(f"积分过程出错 at j={j}, r={r:.6e}: {str(e)}")
                raise RuntimeError(f"积分不稳定 at j={j}, r={r}")

        # 变换回u(r)，进行稳定性检查
        exp_factor = np.exp(self.delta * self.grid.j / 2)
        # 限制指数因子的增长
        exp_factor = np.minimum(exp_factor, 1e30)

        u = v * exp_factor

        # 最终的数值检查
        if np.any(np.isnan(u)) or np.any(np.abs(u) > 1e20):
            logger.warning("最终波函数包含不稳定值，尝试归一化处理")
            # 对不稳定值进行处理
            u = np.nan_to_num(u, nan=0.0, posinf=1e20, neginf=-1e20)
            # 简单归一化
            norm = np.sqrt(np.sum(u * u) * self.delta)
            if norm > 0:
                u /= norm

        return u

    def shooting_solve(
        self, E_min: float, E_max: float, target_nodes: int
    ) -> Tuple[float, np.ndarray]:
        """打靶法求解本征值和本征函数

        Args:
            E_min, E_max: 能量搜索范围
            target_nodes: 目标节点数

        Returns:
            (E, u): 能量本征值和对应的波函数
        """

        def objective(E: float) -> float:
            """目标函数：计算r=0处的值并考虑节点数"""
            u = self.integrate_inward(E)
            nodes = WavefunctionTools.count_nodes(u)

            if nodes != target_nodes:
                return 1e3 * (nodes - target_nodes)  # 节点数不对，施加惩罚
            return u[0]  # 返回r=0处的值

        try:
            # 使用更宽松的收敛条件
            result = root_scalar(
                objective,
                method="brentq",  # 改用更稳定的brentq方法
                bracket=[E_min, E_max],  # 使用bracket代替x0,x1
                rtol=1e-6,  # 放宽收敛条件
                maxiter=1000,  # 增加最大迭代次数
            )

            if not result.converged:
                # 尝试不同的初始能量范围
                E_min_new = E_min * 1.5
                E_max_new = E_max * 0.5
                result = root_scalar(
                    objective,
                    method="brentq",
                    bracket=[E_min_new, E_max_new],
                    rtol=1e-6,
                    maxiter=1000,
                )

        except ValueError as e:
            logger.error(f"能量求解出错: {str(e)}")
            raise RuntimeError("能量求解失败，请检查能量范围和初始条件")

        if not result.converged:
            raise RuntimeError("能量求解未收敛!")

        # 计算最终波函数
        E = result.root
        u = self.integrate_inward(E)

        return E, u


# Part 5 FiniteDifferenceSolver
class FiniteDifferenceSolver:
    """有限差分法求解器"""

    def __init__(self, grid: RadialGrid, V: callable, l: int):
        """初始化有限差分法求解器

        Args:
            grid: 径向网格对象
            V: 势能函数
            l: 角量子数
        """
        self.grid = grid
        self.V = V
        self.l = l

    def construct_hamiltonian(self) -> spmatrix:
        """构建哈密顿量矩阵

        非均匀网格上的哈密顿量包括:
        1. 动能项（二阶导数）
        2. 势能项
        3. 向心势项

        Returns:
            哈密顿量稀疏矩阵
        """
        N = len(self.grid.r)
        delta = self.grid.config.delta

        # 非均匀网格上的二阶导数项
        d2_coef = np.exp(-2 * delta * self.grid.j) / (delta**2)
        d1_coef = -np.exp(-delta * self.grid.j) / (2 * delta)

        # 势能和角动量项
        diag = (
            -2 * d2_coef
            + self.V(self.grid.r)
            + self.l * (self.l + 1) / (2 * self.grid.r**2)
        )

        # 非对角项
        upper_diag = d2_coef[:-1] + d1_coef[:-1]
        lower_diag = d2_coef[:-1] - d1_coef[:-1]

        # 构建稀疏矩阵
        H = diags([lower_diag, diag, upper_diag], [-1, 0, 1], format="csr")

        # 边界条件
        H[0, :] = 0
        H[0, 0] = 1.0
        H[-1, :] = 0
        H[-1, -1] = 1.0

        return H

    def solve(self, n_states: int) -> Tuple[np.ndarray, np.ndarray]:
        """求解本征值问题

        Args:
            n_states: 需要求解的本征态数量

        Returns:
            (energies, states): 本征能量和本征态
        """
        # 构建哈密顿量
        H = self.construct_hamiltonian()

        # 求解本征值问题
        energies, states = linalg.eigsh(
            H, k=n_states, which="SA", return_eigenvectors=True
        )

        # 排序
        idx = np.argsort(energies)
        energies = energies[idx]
        states = states[:, idx]

        # 归一化
        for i in range(states.shape[1]):
            norm = np.sqrt(np.trapz(states[:, i] ** 2, self.grid.r))
            states[:, i] /= norm

        return energies, states


# Part 6 Results and Analysis
class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, grid: RadialGrid):
        """初始化可视化器

        Args:
            grid: 径向网格对象
        """
        self.grid = grid

        # 设置matplotlib的基本样式
        plt.style.use("default")

        # 尝试使用seaborn增强可视化效果
        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid", font_scale=1.2)
            self.use_seaborn = True
        except ImportError:
            logger.warning("未能导入seaborn包，将使用matplotlib基本样式")
            self.use_seaborn = False

        # 设置默认的图形大小和DPI
        plt.rcParams["figure.figsize"] = [10.0, 6.0]
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100

        # 设置字体样式
        plt.rcParams["font.family"] = "DejaVu Sans"
        plt.rcParams["mathtext.fontset"] = "dejavusans"

        # 设置轴标签大小
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14

    def plot_wavefunction(
        self,
        u: np.ndarray,
        R: np.ndarray,
        E: float,
        n: int,
        l: int,
        V_type: str,
        R_analytic: Optional[np.ndarray] = None,
    ) -> None:
        """绘制波函数

        Args:
            u: 径向波函数u(r)
            R: 径向波函数R(r)
            E: 能量本征值
            n, l: 量子数
            V_type: 势能类型
            R_analytic: 解析解(如果有)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 上图：波函数
        ax1.plot(self.grid.r, R, "b-", label="R(r) 数值解")
        if R_analytic is not None:
            ax1.plot(self.grid.r, R_analytic, "r--", label="R(r) 解析解")
        ax1.plot(self.grid.r, u, "g:", label="u(r)")
        ax1.set_xlabel("r (Bohr)")
        ax1.set_ylabel("波函数")
        ax1.grid(True)
        ax1.legend()

        # 下图：概率密度
        probability = R * R * self.grid.r * self.grid.r
        ax2.plot(self.grid.r, probability, "b-", label="概率密度 r²R²(r)")
        if R_analytic is not None:
            prob_analytic = R_analytic * R_analytic * self.grid.r * self.grid.r
            ax2.plot(self.grid.r, prob_analytic, "r--", label="解析解概率密度")
        ax2.set_xlabel("r (Bohr)")
        ax2.set_ylabel("概率密度")
        ax2.grid(True)
        ax2.legend()

        # 总标题
        title = f"{V_type}原子: n={n}, l={l}, E={E:.6f} Hartree"
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def plot_energy_scan(
        self, E_array: np.ndarray, u0_array: np.ndarray, n: int, l: int, V_type: str
    ) -> None:
        """绘制能量扫描结果

        Args:
            E_array: 能量数组
            u0_array: 对应的u(0)值
            n, l: 量子数
            V_type: 势能类型
        """
        plt.figure(figsize=(10, 6))
        plt.plot(E_array, u0_array, "b-")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("能量 (Hartree)")
        plt.ylabel("u(r=0)")
        plt.title(f"{V_type}原子能量扫描 (n={n}, l={l})")
        plt.grid(True)
        plt.show()

    def plot_convergence_study(self, results: dict) -> None:
        """绘制收敛性研究结果

        Args:
            results: 收敛性分析结果字典
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(results["delta_h"], results["errors"], "bo-")

        # 添加参考线
        h = np.array(results["delta_h"])
        plt.loglog(
            h, h**2 * results["errors"][0] / h[0] ** 2, "r--", label="O(h²) 参考线"
        )
        plt.loglog(
            h, h**4 * results["errors"][0] / h[0] ** 4, "g--", label="O(h⁴) 参考线"
        )

        plt.xlabel("网格间距 h (log)")
        plt.ylabel("相对误差 % (log)")
        plt.title("收敛性分析")
        plt.legend()
        plt.grid(True)
        plt.show()


class ResultAnalyzer:
    """结果分析类"""

    def __init__(self, grid: RadialGrid, energy_analyzer: EnergyAnalyzer):
        """初始化分析器

        Args:
            grid: 径向网格对象
            energy_analyzer: 能量分析器实例
        """
        self.grid = grid
        self.energy_analyzer = energy_analyzer

        # 设置中文字体
        plt.rcParams["font.family"] = ["Arial Unicode MS", "SimHei"]

        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid")
            self.use_seaborn = True
        except ImportError:
            logger.warning("未能导入seaborn包，将使用matplotlib基本样式")
            self.use_seaborn = False

        # 设置图形样式
        plt.rcParams["figure.figsize"] = [10.0, 6.0]
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["savefig.dpi"] = 100

    def analyze_state(
        self, u: np.ndarray, R: np.ndarray, E: float, n: int, l: int, V_type: str
    ) -> dict:
        """分析量子态的性质

        Args:
            u, R: 波函数
            E: 能量
            n, l: 量子数
            V_type: 势能类型

        Returns:
            分析结果字典
        """
        # 基本性质
        results = {
            "quantum_numbers": {"n": n, "l": l},
            "energy": E,
            "nodes": WavefunctionTools.count_nodes(u),
            "max_probability_r": self.grid.r[
                np.argmax(R * R * self.grid.r * self.grid.r)
            ],
            "normalization": np.trapz(R * R * self.grid.r * self.grid.r, self.grid.r),
        }

        # 能量分析
        energy_analysis = self.energy_analyzer.compare_with_theory(E, V_type, n, l)
        results.update({"energy_analysis": energy_analysis})

        # 期望值
        results.update(
            {
                "expectation_values": {
                    "r": self._expectation_r(R),
                    "r2": self._expectation_r2(R),
                    "kinetic": self._expectation_kinetic(u),
                }
            }
        )

        return results

    def _expectation_r(self, R: np.ndarray) -> float:
        """计算<r>"""
        return np.trapz(R * R * self.grid.r**3, self.grid.r)

    def _expectation_r2(self, R: np.ndarray) -> float:
        """计算<r²>"""
        return np.trapz(R * R * self.grid.r**4, self.grid.r)

    def _expectation_kinetic(self, u: np.ndarray) -> float:
        """计算动能期望值"""
        # 数值微分计算d²u/dr²
        du = np.gradient(u, self.grid.r)
        d2u = np.gradient(du, self.grid.r)
        return -0.5 * np.trapz(u * d2u, self.grid.r)

    def print_analysis(self, results: dict) -> None:
        """打印分析结果

        Args:
            results: analyze_state返回的结果字典
        """
        print("\n量子态分析:")
        print(
            f"量子数: n={results['quantum_numbers']['n']}, "
            f"l={results['quantum_numbers']['l']}"
        )
        print(f"能量: E = {results['energy']:.6f} Hartree")

        ea = results["energy_analysis"]
        if ea["theoretical_E"] is not None:
            print(f"理论值: E = {ea['theoretical_E']:.6f} Hartree")
            print(f"相对误差: {ea['relative_error']:.6f}%")

        print(f"\n波函数性质:")
        print(f"节点数: {results['nodes']}")
        print(f"最大概率密度位置: r = {results['max_probability_r']:.4f} Bohr")
        print(f"归一化检查: {results['normalization']:.6f}")

        ev = results["expectation_values"]
        print(f"\n期望值:")
        print(f"<r> = {ev['r']:.4f} Bohr")
        print(f"<r²> = {ev['r2']:.4f} Bohr²")
        print(f"<T> = {ev['kinetic']:.4f} Hartree")

    def create_convergence_study(self, n_values: List[int]) -> dict:
        """进行收敛性分析

        Args:
            n_values: 网格点数列表

        Returns:
            收敛性分析结果字典
        """
        results = {"n_points": n_values, "energies": [], "errors": [], "delta_h": []}

        # 基准解（使用最密的网格）
        config_ref = copy.deepcopy(self.grid.config)
        config_ref.j_max = max(n_values)
        solver_ref = ShootingSolver(
            RadialGrid(config_ref),
            PotentialFunction.get_potential(config_ref.V_type),
            config_ref.l,
        )
        E_ref, _ = solver_ref.shooting_solve(
            -1.0, -0.1, config_ref.n - config_ref.l - 1  # 能量搜索范围  # 目标节点数
        )

        # 对不同网格点数求解
        for n in n_values:
            config = copy.deepcopy(self.grid.config)
            config.j_max = n
            solver = ShootingSolver(
                RadialGrid(config),
                PotentialFunction.get_potential(config.V_type),
                config.l,
            )

            try:
                E, _ = solver.shooting_solve(-1.0, -0.1, config.n - config.l - 1)

                results["energies"].append(E)
                results["errors"].append(abs((E - E_ref) / E_ref) * 100)
                results["delta_h"].append(config.r_max / n)

            except Exception as e:
                logger.warning(f"网格点数 {n} 求解失败: {str(e)}")
                continue

        return results


# Part 7 Main Workflow
class RadialSchrodingerSolver:
    """径向薛定谔方程主求解器类"""

    def __init__(self, config: SolverConfig):
        """初始化求解器

        Args:
            config: 求解器配置
        """
        self.config = config
        # 初始化各个组件
        self.grid = RadialGrid(config)
        self.V = PotentialFunction.get_potential(config.V_type)
        self.wave_processor = WavefunctionProcessor(self.grid.r, config.l)
        self.energy_analyzer = EnergyAnalyzer(get_theoretical_values())
        self.visualizer = ResultVisualizer(self.grid)
        self.analyzer = ResultAnalyzer(self.grid, self.energy_analyzer)

        # 根据方法选择求解器
        if config.method == "shooting":
            self.solver = ShootingSolver(self.grid, self.V, config.l)
        else:
            self.solver = FiniteDifferenceSolver(self.grid, self.V, config.l)

        logger.info(
            f"初始化{config.V_type}原子求解器: "
            f"n={config.n}, l={config.l}, 方法={config.method}"
        )

    def solve_state(self) -> dict:
        """求解量子态

        Returns:
            计算结果字典
        """
        try:
            # 计算能量范围
            E_min, E_max = get_initial_energy_range(self.config)

            # 能量扫描
            logger.info("开始能量扫描...")
            E_scan = np.linspace(E_min, E_max, 100)
            u0_array = []
            for E in E_scan:
                if self.config.method == "shooting":
                    u = self.solver.integrate_inward(E)
                    u0_array.append(u[0])
                else:
                    # 有限差分法不需要扫描
                    break

            # 求解本征值和本征函数
            logger.info("求解本征值问题...")
            if self.config.method == "shooting":
                E, u = self.solver.shooting_solve(
                    E_min, E_max, self.config.n - self.config.l - 1
                )
                u_norm, R = self.wave_processor.normalize_wavefunction(u)
                states = {"energy": E, "u": u_norm, "R": R}
            else:
                energies, wavefunctions = self.solver.solve(self.config.n_states)
                states = {"energies": energies, "states": wavefunctions}

            # 获取解析解(如果有)
            R_analytic = None
            if self.config.V_type == "hydrogen":
                R_analytic = WavefunctionTools.get_analytic_hydrogen(
                    self.grid.r, self.config.n, self.config.l
                )

            # 分析结果
            logger.info("分析结果...")
            if self.config.method == "shooting":
                analysis = self.analyzer.analyze_state(
                    states["u"],
                    states["R"],
                    states["energy"],
                    self.config.n,
                    self.config.l,
                    self.config.V_type,
                )
            else:
                analysis = []
                for i, (E, state) in enumerate(
                    zip(states["energies"], states["states"].T)
                ):
                    u_norm, R = self.wave_processor.normalize_wavefunction(state)
                    analysis.append(
                        self.analyzer.analyze_state(
                            u_norm,
                            R,
                            E,
                            i + self.config.l + 1,
                            self.config.l,
                            self.config.V_type,
                        )
                    )

            # 可视化
            logger.info("绘制结果...")
            if self.config.method == "shooting":
                # 绘制能量扫描
                self.visualizer.plot_energy_scan(
                    E_scan, u0_array, self.config.n, self.config.l, self.config.V_type
                )
                # 绘制波函数
                self.visualizer.plot_wavefunction(
                    states["u"],
                    states["R"],
                    states["energy"],
                    self.config.n,
                    self.config.l,
                    self.config.V_type,
                    R_analytic,
                )
            else:
                for i, (E, state) in enumerate(
                    zip(states["energies"], states["states"].T)
                ):
                    u_norm, R = self.wave_processor.normalize_wavefunction(state)
                    self.visualizer.plot_wavefunction(
                        u_norm,
                        R,
                        E,
                        i + self.config.l + 1,
                        self.config.l,
                        self.config.V_type,
                    )

            return {
                "states": states,
                "analysis": analysis,
                "grid": self.grid.get_grid_info(),
            }

        except Exception as e:
            logger.error(f"计算过程出错: {str(e)}")
            raise


def run_examples():
    """运行示例计算"""
    # 配置示例
    configs = [
        # 氢原子基态
        SolverConfig(
            V_type="hydrogen", n=1, l=0, method="shooting", r_max=30.0, j_max=1000
        ),
        # 氢原子第一激发态
        SolverConfig(
            V_type="hydrogen", n=2, l=0, method="shooting", r_max=30.0, j_max=1000
        ),
        # 氢原子2p态
        SolverConfig(
            V_type="hydrogen", n=2, l=1, method="shooting", r_max=30.0, j_max=1000
        ),
        # 锂原子基态
        SolverConfig(
            V_type="lithium", n=1, l=0, method="shooting", r_max=30.0, j_max=1000
        ),
        # 锂原子2p态
        SolverConfig(
            V_type="lithium", n=2, l=1, method="shooting", r_max=30.0, j_max=1000
        ),
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"求解 {config.V_type}原子态 n={config.n}, l={config.l}")
        print(f"使用{config.method}方法")
        print("=" * 60)

        # 创建求解器并求解
        solver = RadialSchrodingerSolver(config)
        results = solver.solve_state()

        # 展示分析结果
        if isinstance(results["analysis"], list):
            for analysis in results["analysis"]:
                solver.analyzer.print_analysis(analysis)
        else:
            solver.analyzer.print_analysis(results["analysis"])

        # 对基态进行收敛性研究
        if config.n == 1 and config.l == 0:
            print("\n进行收敛性研究...")
            n_values = [100, 200, 500, 1000, 2000]
            solver.visualizer.plot_convergence_study(
                solver.analyzer.create_convergence_study(n_values)
            )


if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        run_examples()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {str(e)}")
        raise
