"""径向薛定谔方程求解器的可视化模块

负责绘制波函数、概率密度、能量扫描结果等图像。
提供结果可视化和分析展示功能。

Classes:
   ResultVisualizer: 结果可视化类
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ResultVisualizer:
    """结果可视化类"""

    def __init__(self, r: np.ndarray):
        """初始化可视化器

        Parameters
        ----------
        r : np.ndarray
            径向网格点
        """
        self.r = r
        self._setup_style()

    def _setup_style(self):
        """设置绘图样式"""
        plt.style.use("default")

        # 尝试使用seaborn提升样式
        try:
            import seaborn as sns

            sns.set_theme(style="whitegrid", font_scale=1.2)
        except ImportError:
            logger.info("未能导入seaborn包，使用matplotlib基本样式")

        # 检测系统
        import platform

        system = platform.system()

        # 根据操作系统设置中文字体
        if system == "Darwin":  # macOS
            plt.rcParams["font.family"] = ["Arial Unicode MS"]
        elif system == "Windows":
            plt.rcParams["font.family"] = ["Microsoft YaHei"]
        elif system == "Linux":
            plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]

        # 如果上述字体都不可用，尝试使用系统默认字体
        try:
            plt.rcParams["font.sans-serif"] = [
                "Arial Unicode MS",
                "SimSun",
                "STSong",
                "SimHei",
            ] + plt.rcParams["font.sans-serif"]
        except:
            logger.warning("未能设置理想的中文字体，尝试使用系统默认字体")

        # 设置其他参数
        plt.rcParams.update(
            {
                "figure.figsize": [10.0, 6.0],
                "figure.dpi": 100,
                "savefig.dpi": 100,
                "axes.labelsize": 12,
                "axes.titlesize": 14,
                "axes.unicode_minus": False,  # 解决负号显示问题
            }
        )

    def plot_wavefunction(
        self,
        u: np.ndarray,
        R: np.ndarray,
        E: float,
        n: int,
        l: int,
        V_type: str,
        R_analytic: Optional[np.ndarray] = None,
    ):
        """绘制波函数

        Parameters
        ----------
        u : np.ndarray
            径向波函数u(r)
        R : np.ndarray
            径向波函数R(r)
        E : float
            能量本征值
        n : int
            主量子数
        l : int
            角量子数
        V_type : str
            势能类型
        R_analytic : np.ndarray, optional
            解析解(如果有)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 上图：波函数
        ax1.plot(self.r, R, "b-", label="R(r) 数值解")
        if R_analytic is not None:
            ax1.plot(self.r, R_analytic, "r--", label="R(r) 解析解")
        ax1.plot(self.r, u, "g:", label="u(r)")
        ax1.set_xlabel("r (Bohr)")
        ax1.set_ylabel("波函数")
        ax1.grid(True)
        ax1.legend()

        # 下图：概率密度
        probability = R * R * self.r * self.r
        ax2.plot(self.r, probability, "b-", label="概率密度 r²R²(r)")
        if R_analytic is not None:
            prob_analytic = R_analytic * R_analytic * self.r * self.r
            ax2.plot(self.r, prob_analytic, "r--", label="解析解概率密度")
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
    ):
        """绘制能量扫描结果

        Parameters
        ----------
        E_array : np.ndarray
            能量数组
        u0_array : np.ndarray
            对应的u(0)值
        n : int
            主量子数
        l : int
            角量子数
        V_type : str
            势能类型
        """
        plt.figure(figsize=(10, 6))
        plt.plot(E_array, u0_array, "b-")
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("能量 (Hartree)")
        plt.ylabel("u(r=0)")
        plt.title(f"{V_type}原子能量扫描 (n={n}, l={l})")
        plt.grid(True)
        plt.show()

    def plot_convergence_study(self, results: Dict):
        """绘制收敛性研究结果

        Parameters
        ----------
        results : Dict
            收敛性分析结果字典
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
