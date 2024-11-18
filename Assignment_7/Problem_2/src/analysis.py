"""径向薛定谔方程的波函数分析和可视化"""

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple, List
import logging

from base import SolverConfig, RadialGrid

logger = logging.getLogger(__name__)


class WavefunctionAnalyzer:
    """波函数分析工具
    
    Parameters
    ----------
    grid : RadialGrid
        径向网格对象
    l : int
        角量子数
    """
    
    def __init__(self, grid: RadialGrid, l: int):
        self.grid = grid
        self.l = l

    def normalize(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """归一化波函数

        Parameters
        ----------
        u : ndarray
            原始波函数u(r)

        Returns
        -------
        ndarray
            归一化的u(r)
        ndarray
            归一化的R(r) = u(r)/r

        Notes
        -----
        计算归一化常数: N = [∫|u(r)|²dr]^(-1/2)
        """
        # 数值稳定性检查
        if np.any(np.isnan(u)) or np.any(np.isinf(u)):
            logger.warning("波函数包含无效值")
            return u, u/self.grid.r

        # 处理可能的零值
        mask = np.abs(u) > 1e-15
        if not np.any(mask):
            logger.warning("波函数全为零")
            return u, np.zeros_like(u)

        # 计算归一化常数
        norm_u = np.sqrt(np.trapz(u[mask]**2, self.grid.r[mask]))
        if norm_u < 1e-15:
            logger.warning("归一化常数接近零")
            return u, np.zeros_like(u)

        u_norm = u/norm_u

        # 计算R(r)
        R = np.zeros_like(u_norm)
        nonzero_r = self.grid.r > 1e-10
        R[nonzero_r] = u_norm[nonzero_r]/self.grid.r[nonzero_r]

        # r=0处的处理
        if self.l == 0:
            # l=0时，使用临近点估计R(0)
            if len(R[nonzero_r]) > 0:
                R[0] = R[nonzero_r][0]
        else:
            # l>0时，R(0)=0
            R[0] = 0

        return u_norm, R

    def get_analytic_hydrogen(self, n: int) -> Optional[np.ndarray]:
        """氢原子解析解

        Parameters
        ----------
        n : int
            主量子数

        Returns
        -------
        ndarray or None
            解析波函数R(r), 无解析解时返回None

        Notes
        -----
        目前支持n≤3的s,p,d态
        """
        r = self.grid.r
        l = self.l

        # 归一化常数
        norm = np.sqrt(4 * factorial(n - l - 1)/(n**4 * factorial(n + l)))

        if n == 1 and l == 0:   # 1s
            return norm * 2 * np.exp(-r)
        elif n == 2:
            if l == 0:          # 2s
                return norm * (2 - r) * np.exp(-r/2)
            elif l == 1:        # 2p
                return norm * r * np.exp(-r/2)
        elif n == 3:
            if l == 0:          # 3s
                return norm * (27 - 18*r + 2*r**2) * np.exp(-r/3)
            elif l == 1:        # 3p
                return norm * r * (6 - r) * np.exp(-r/3)
            elif l == 2:        # 3d
                return norm * r**2 * np.exp(-r/3)
        return None

    def analyze_state(self, u: np.ndarray, R: np.ndarray, E: float) -> Dict:
        """分析量子态性质

        Parameters
        ----------
        u : ndarray
            归一化的u(r)
        R : ndarray
            归一化的R(r)
        E : float
            本征能量

        Returns
        -------
        dict
            包含各种物理量的字典:
            - 节点数
            - 最大概率位置
            - 期望值<r>, <r²>
            - 动能期望值
            - 不确定度Δr
        """
        r = self.grid.r
        
        # 基本性质
        nodes = self._count_nodes(u)
        prob_density = R**2 * r**2
        r_max_prob = r[np.argmax(prob_density)]
        
        # 计算期望值
        r_mean = np.trapz(prob_density * r, r)
        r2_mean = np.trapz(prob_density * r**2, r)
        delta_r = np.sqrt(r2_mean - r_mean**2)

        # 计算动能期望值
        du = np.gradient(u, r)
        d2u = np.gradient(du, r)
        T_mean = -0.5 * np.trapz(u * d2u, r)

        return {
            "energy": E,
            "nodes": nodes,
            "r_max_prob": r_max_prob,
            "r_mean": r_mean,
            "r2_mean": r2_mean,
            "delta_r": delta_r,
            "T_mean": T_mean,
        }

    @staticmethod
    def _count_nodes(u: np.ndarray, eps: float = 1e-10) -> int:
        """计算波函数节点数

        Parameters
        ----------
        u : ndarray
            波函数数组
        eps : float, optional
            判定零点的阈值, 默认1e-10

        Returns
        -------
        int
            节点数
        """
        u_filtered = np.where(abs(u) < eps, 0, u)
        return len(np.where(np.diff(np.signbit(u_filtered)))[0])


class ResultVisualizer:
    """结果可视化工具
    
    Parameters
    ----------
    grid : RadialGrid
        径向网格对象
    """
    
    def __init__(self, grid: RadialGrid):
        self.grid = grid
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图样式"""
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': [10.0, 6.0],
            'figure.dpi': 100,
            'savefig.dpi': 100,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12
        })

    def plot_wavefunction(self, u: np.ndarray, R: np.ndarray, E: float,
                       n: int, l: int, R_analytic: Optional[np.ndarray] = None):
        """绘制波函数

        Parameters
        ----------
        u : ndarray
            径向波函数u(r)
        R : ndarray
            径向波函数R(r)
        E : float
            能量本征值
        n : int
            主量子数
        l : int
            角量子数
        R_analytic : ndarray, optional
            解析解(如果有)
        """
        r = self.grid.r
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        # 波函数图
        ax1.plot(r, R, 'b-', label='R(r) 数值解')
        if R_analytic is not None:
            ax1.plot(r, R_analytic, 'r--', label='R(r) 解析解')
        ax1.plot(r, u, 'g:', label='u(r)')
        ax1.set_xlabel('r (Bohr)')
        ax1.set_ylabel('波函数')
        ax1.legend()
        ax1.grid(True)

        # 概率密度图
        prob = R**2 * r**2
        ax2.plot(r, prob, 'b-', label='概率密度 r²R²(r)')
        if R_analytic is not None:
            prob_analytic = R_analytic**2 * r**2
            ax2.plot(r, prob_analytic, 'r--', label='解析解概率密度')
        ax2.set_xlabel('r (Bohr)')
        ax2.set_ylabel('概率密度')
        ax2.legend()
        ax2.grid(True)

        # 设置标题
        title = f'n={n}, l={l}, E={E:.6f} Hartree'
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

    def plot_convergence(self, results: Dict):
        """绘制收敛性分析结果

        Parameters
        ----------
        results : dict
            收敛性分析结果字典
        """
        plt.figure(figsize=(10, 6))
        plt.loglog(results['delta_h'], results['errors'], 'bo-')
        
        h = np.array(results['delta_h'])
        plt.loglog(h, h**2 * results['errors'][0]/h[0]**2, 
                  'r--', label='O(h²)')
        plt.loglog(h, h**4 * results['errors'][0]/h[0]**4, 
                  'g--', label='O(h⁴)')

        plt.xlabel('网格间距 h (log)')
        plt.ylabel('相对误差 % (log)')
        plt.title('收敛性分析')
        plt.legend()
        plt.grid(True)
