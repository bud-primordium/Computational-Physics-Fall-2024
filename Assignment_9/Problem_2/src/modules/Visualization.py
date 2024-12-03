"""
改进的海森堡模型可视化功能
包含簇的颜色显示和动画
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Set, Dict, Tuple, Optional
from HeisenbergFCC import HeisenbergFCC
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import colorsys
import platform


def configure_matplotlib_fonts():
    """配置matplotlib的字体设置"""
    system = platform.system()
    if system == "Darwin":  # macOS
        plt.rcParams["font.family"] = ["Arial Unicode MS"]
    elif system == "Windows":
        plt.rcParams["font.family"] = ["Microsoft YaHei"]
    else:  # Linux
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
    # 备用字体
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


class Visualization:
    def __init__(self):
        self.colormap = plt.cm.viridis
        # 为自旋和簇添加颜色映射
        self.spin_colors = {
            "arrows": "royalblue",
            "points": "gray",
            "clusters": self._generate_cluster_colors(20),  # 预生成20种不同的颜色
        }
        configure_matplotlib_fonts()

    def _generate_cluster_colors(self, n: int) -> List[str]:
        """生成n种视觉上易区分的颜色"""
        colors = []
        for i in range(n):
            hue = i / n
            saturation = 0.7 + 0.3 * (i % 2)  # 交替使用不同饱和度
            value = 0.8 + 0.2 * (i % 2)  # 交替使用不同明度
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        return colors

    def _get_cluster_colors(self, clusters: List[Set[int]]) -> Dict[int, str]:
        """为每个簇分配颜色"""
        colors = {}
        for i, cluster in enumerate(clusters):
            color = self.spin_colors["clusters"][i % len(self.spin_colors["clusters"])]
            for spin_idx in cluster:
                colors[spin_idx] = color
        return colors

    def plot_spins(
        self, model: HeisenbergFCC, clusters: Optional[List[Set[int]]] = None, ax=None
    ) -> None:
        """
        绘制自旋构型，可选择性地显示簇

        参数:
            model: HeisenbergFCC模型实例
            clusters: 可选的簇列表
            ax: 可选的 Matplotlib Axes 对象
        """
        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax.clear()

        coords = np.array([coord for coord in model._get_index_to_coord().values()])
        spins = model.spins

        # 如果提供了簇信息，使用不同颜色显示
        if clusters:
            cluster_colors = self._get_cluster_colors(clusters)
            colors = [
                cluster_colors.get(i, self.spin_colors["arrows"])
                for i in range(len(spins))
            ]
        else:
            colors = [self.spin_colors["arrows"]] * len(spins)

        # 绘制格点
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=self.spin_colors["points"],
            alpha=0.3,
        )

        # 绘制自旋箭头
        for coord, spin, color in zip(coords, spins, colors):
            ax.quiver(
                coord[0],
                coord[1],
                coord[2],
                spin[0],
                spin[1],
                spin[2],
                color=color,
                length=0.2,
                normalize=True,
                alpha=0.8,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Spin Configuration" + (" with Clusters" if clusters else ""))

        if ax is None:
            plt.show()

    def animate_update(
        self, model: HeisenbergFCC, updater, steps: int = 100
    ) -> FuncAnimation:
        """改进的动画函数，显示簇的更新"""
        fig = plt.figure(figsize=(12, 10))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 4, 1])

        title_ax = fig.add_subplot(gs[0])
        title_ax.axis("off")
        title = title_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=12)

        ax = fig.add_subplot(gs[1], projection="3d")
        coords = np.array([coord for coord in model._get_index_to_coord().values()])

        # 存储所有箭头对象
        arrows = []

        info_ax = fig.add_subplot(gs[2])
        info_ax.axis("off")
        info = info_ax.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

        def init():
            ax.clear()
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=self.spin_colors["points"],
                alpha=0.3,
                s=50,
            )

            # 初始化箭头
            arrows.clear()
            for coord, spin in zip(coords, model.spins):
                arrow = ax.quiver(
                    coord[0],
                    coord[1],
                    coord[2],
                    spin[0],
                    spin[1],
                    spin[2],
                    color=self.spin_colors["arrows"],
                    length=0.3,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    linewidth=1.5,
                    alpha=0.8,
                )
                arrows.append(arrow)

            ax.set_xlim(-0.5, model.L + 0.5)
            ax.set_ylim(-0.5, model.L + 0.5)
            ax.set_zlim(-0.5, model.L + 0.5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            title.set_text(f"FCC Heisenberg Model (L={model.L}, T={model.T:.2f})")
            return arrows + [title, info]

        def update(frame):
            E_before = model.energy / model.N
            M_before = np.linalg.norm(model.calculate_magnetization())

            # 执行更新
            updater.update()

            E_after = model.energy / model.N
            M_after = np.linalg.norm(model.calculate_magnetization())

            # 清除旧箭头
            for arrow in arrows:
                arrow.remove()
            arrows.clear()

            # 获取簇信息并分配颜色
            colors = [self.spin_colors["arrows"]] * model.N
            if hasattr(updater, "cluster"):
                cluster = updater.cluster
                cluster_color = self.spin_colors["clusters"][
                    frame % len(self.spin_colors["clusters"])
                ]
                for idx in cluster:
                    colors[idx] = cluster_color
            elif hasattr(updater, "clusters"):
                cluster_colors = self._get_cluster_colors(updater.clusters)
                colors = [
                    cluster_colors.get(i, self.spin_colors["arrows"])
                    for i in range(model.N)
                ]

            # 绘制新箭头
            for coord, spin, color in zip(coords, model.spins, colors):
                arrow = ax.quiver(
                    coord[0],
                    coord[1],
                    coord[2],
                    spin[0],
                    spin[1],
                    spin[2],
                    color=color,
                    length=0.3,
                    normalize=True,
                    arrow_length_ratio=0.3,
                    linewidth=1.5,
                    alpha=0.8,
                )
                arrows.append(arrow)

            # 更新信息显示
            info_text = (
                f"Step: {frame}\n"
                f"Energy/N: {E_after:.4f} (ΔE/N: {E_after-E_before:.4f})\n"
                f"Magnetization: {M_after:.4f} (ΔM: {M_after-M_before:.4f})"
            )

            if hasattr(updater, "cluster"):
                info_text += f"\nCluster Size: {len(updater.cluster)}"
            elif hasattr(updater, "clusters"):
                total_size = sum(len(c) for c in updater.clusters)
                info_text += f"\nNumber of Clusters: {len(updater.clusters)}"
                info_text += f"\nTotal Spins in Clusters: {total_size}"

            info.set_text(info_text)
            return arrows + [title, info]

        plt.tight_layout()

        ani = FuncAnimation(
            fig, update, frames=steps, init_func=init, interval=200, blit=False
        )
        return ani

    def plot_correlation(self, corr: np.ndarray, distance: np.ndarray) -> None:
        """绘制关联函数"""
        plt.figure(figsize=(8, 6))
        plt.plot(distance, corr, "o-")
        plt.xlabel("Distance")
        plt.ylabel("Correlation")
        plt.yscale("log")
        plt.title("Spin-Spin Correlation Function")
        plt.grid(True)
        plt.show()

    def plot_structure_factor(self, sf: np.ndarray) -> None:
        """绘制结构因子"""
        L = sf.shape[0]
        extent = [-np.pi, np.pi, -np.pi, np.pi]

        plt.figure(figsize=(8, 6))
        plt.imshow(sf[:, :, L // 2], extent=extent, cmap="hot")
        plt.colorbar(label="S(q)")
        plt.xlabel("qx")
        plt.ylabel("qy")
        plt.title("Structure Factor S(q) [qz=0 plane]")
        plt.show()

    def plot_physical_quantities(
        self, results: Dict[Tuple[int, float, str], List]
    ) -> None:
        """绘制物理量随温度的变化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        L_values = sorted(set(L for (L, _, _) in results.keys()))
        T_values = sorted(set(T for (_, T, _) in results.keys()))

        for L in L_values:
            E = [np.mean([m.E for m in results[(L, T, "wolff")]]) for T in T_values]
            M = [np.mean([m.M for m in results[(L, T, "wolff")]]) for T in T_values]
            C = [
                np.mean([m.specific_heat for m in results[(L, T, "wolff")]])
                for T in T_values
            ]
            chi = [
                np.mean([m.susceptibility for m in results[(L, T, "wolff")]])
                for T in T_values
            ]

            ax1.plot(T_values, E, "o-", label=f"L={L}")
            ax2.plot(T_values, M, "o-", label=f"L={L}")
            ax3.plot(T_values, C, "o-", label=f"L={L}")
            ax4.plot(T_values, chi, "o-", label=f"L={L}")

        ax1.set_xlabel("Temperature")
        ax1.set_ylabel("Energy")
        ax1.legend()
        ax1.grid(True)

        ax2.set_xlabel("Temperature")
        ax2.set_ylabel("Magnetization")
        ax2.legend()
        ax2.grid(True)

        ax3.set_xlabel("Temperature")
        ax3.set_ylabel("Specific Heat")
        ax3.legend()
        ax3.grid(True)

        ax4.set_xlabel("Temperature")
        ax4.set_ylabel("Susceptibility")
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.show()
