"""
海森堡模型的交互式模拟控制类
包含温度控制、更新方法选择、动画控制等功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button, Slider
from matplotlib.animation import FuncAnimation
import threading
import time
from typing import Optional, List, Dict
from HeisenbergFCC import HeisenbergFCC
from updaters import create_updater
from Visualization import Visualization


class InteractiveSimulation:
    def __init__(self, model: HeisenbergFCC):
        self.model = model
        self.vis = Visualization()
        self.updater = create_updater(self.model, "single")
        self.animation: Optional[FuncAnimation] = None
        self.running = False
        self.update_thread: Optional[threading.Thread] = None

        # 初始化界面
        self.setup_interface()

    def setup_interface(self):
        """设置交互界面"""
        # 创建主窗口
        self.fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 4)

        # 3D显示区域
        self.ax_3d = self.fig.add_subplot(gs[:, :3], projection="3d")
        self.setup_3d_view()

        # 控制面板区域
        self.setup_controls(gs)

        # 初始化信息显示
        self.info_text = self.fig.text(0.02, 0.02, "", transform=self.fig.transFigure)

        # 调整布局
        plt.tight_layout()

    def setup_3d_view(self):
        """设置3D视图"""
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")
        self.ax_3d.set_title("FCC Heisenberg Model")

        # 初始化显示
        self.update_display()

    def setup_controls(self, gs):
        """设置控制面板"""
        # 更新方法选择
        update_methods = ["single_spin", "sw", "wolff"]
        radio_ax = self.fig.add_subplot(gs[0, 3])
        radio_ax.set_title("Update Method")
        self.radio = RadioButtons(radio_ax, update_methods)
        self.radio.on_clicked(self.update_method_changed)

        # 温度滑块
        temp_ax = self.fig.add_subplot(gs[1, 3])
        self.temp_slider = Slider(
            temp_ax, "Temperature", 0.1, 5.0, valinit=self.model.T, valstep=0.1
        )
        self.temp_slider.on_changed(self.temperature_changed)

        # 开始/停止按钮
        button_ax = self.fig.add_subplot(gs[2, 3])
        self.button = Button(button_ax, "Start")
        self.button.on_clicked(self.toggle_simulation)

    def update_method_changed(self, label: str):
        """更新方法改变时的回调函数"""
        # 停止当前模拟
        self.stop_simulation()
        # 创建新的更新器
        self.updater = create_updater(self.model, label)
        # 更新显示
        self.update_display()

    def temperature_changed(self, value: float):
        """温度改变时的回调函数"""
        self.model.T = value
        self.model.beta = 1.0 / value
        self.update_display()

    def toggle_simulation(self, event):
        """切换模拟状态"""
        if self.running:
            self.stop_simulation()
            self.button.label.set_text("Start")
        else:
            self.start_simulation()
            self.button.label.set_text("Stop")
        plt.draw()

    def start_simulation(self):
        """启动模拟"""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self.run_simulation)
            self.update_thread.daemon = True
            self.update_thread.start()

    def stop_simulation(self):
        """停止模拟"""
        self.running = False
        if self.update_thread is not None:
            self.update_thread.join()

    def run_simulation(self):
        """运行模拟循环"""
        while self.running:
            # 执行更新
            self.updater.update()
            # 计算物理量
            E = self.model.energy / self.model.N
            M = np.linalg.norm(self.model.calculate_magnetization())

            # 更新信息显示
            info_text = (
                f"Energy/N: {E:.4f}\n"
                f"Magnetization: {M:.4f}\n"
                f"Temperature: {self.model.T:.2f}\n"
            )

            # 添加簇信息
            if hasattr(self.updater, "cluster"):
                info_text += f"Cluster Size: {len(self.updater.cluster)}\n"
            elif hasattr(self.updater, "clusters"):
                total_size = sum(len(c) for c in self.updater.clusters)
                info_text += f"Number of Clusters: {len(self.updater.clusters)}\n"
                info_text += f"Total Spins in Clusters: {total_size}\n"

            # 更新显示
            self.info_text.set_text(info_text)
            self.update_display()

            # 控制更新频率
            time.sleep(0.1)

    def update_display(self):
        """更新显示"""
        self.ax_3d.clear()

        # 获取坐标和自旋
        coords = np.array(
            [coord for coord in self.model._get_index_to_coord().values()]
        )
        spins = self.model.spins

        # 绘制格点
        self.ax_3d.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2], c="gray", alpha=0.3
        )

        # 确定颜色
        colors = ["royalblue"] * len(spins)
        if hasattr(self.updater, "cluster"):
            for idx in self.updater.cluster:
                colors[idx] = "red"
        elif hasattr(self.updater, "clusters"):
            for i, cluster in enumerate(self.updater.clusters):
                color = plt.cm.tab20(i % 20)
                for idx in cluster:
                    colors[idx] = color

        # 绘制自旋箭头
        for coord, spin, color in zip(coords, spins, colors):
            self.ax_3d.quiver(
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

        # 设置视图
        self.ax_3d.set_xlim(-0.5, self.model.L + 0.5)
        self.ax_3d.set_ylim(-0.5, self.model.L + 0.5)
        self.ax_3d.set_zlim(-0.5, self.model.L + 0.5)
        self.ax_3d.set_xlabel("X")
        self.ax_3d.set_ylabel("Y")
        self.ax_3d.set_zlabel("Z")

        plt.draw()

    def show(self):
        """显示界面"""
        plt.show()


if __name__ == "__main__":
    # 主程序
    model = HeisenbergFCC(L=4, T=1.0)
    interactive_sim = InteractiveSimulation(model)
    interactive_sim.show()
