import tkinter as tk
from tkinter import ttk
from Visualization import Visualization
from HeisenbergFCC import HeisenbergFCC
from updaters import create_updater
import numpy as np
import matplotlib.pyplot as plt


class InteractiveApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Heisenberg Model Interactive Visualization")

        # 初始化模型参数
        self.L = 4
        self.T = 1.0
        self.algorithm = "wolff"

        # 创建模型和可视化实例
        self.model = HeisenbergFCC(L=self.L, T=self.T)
        self.updater = create_updater(self.model, self.algorithm)
        self.vis = Visualization()

        # 创建 GUI 控件
        self.create_widgets()

    def create_widgets(self):
        # 创建参数设置区域
        param_frame = tk.Frame(self.master)
        param_frame.pack(side=tk.TOP, fill=tk.X)

        # 系统大小 L 的设置
        tk.Label(param_frame, text="System Size (L):").pack(side=tk.LEFT)
        self.L_var = tk.IntVar(value=self.L)
        L_entry = tk.Entry(param_frame, textvariable=self.L_var, width=5)
        L_entry.pack(side=tk.LEFT)

        # 温度 T 的设置
        tk.Label(param_frame, text="Temperature (T):").pack(side=tk.LEFT)
        self.T_var = tk.DoubleVar(value=self.T)
        T_entry = tk.Entry(param_frame, textvariable=self.T_var, width=5)
        T_entry.pack(side=tk.LEFT)

        # 更新算法的选择
        tk.Label(param_frame, text="Algorithm:").pack(side=tk.LEFT)
        self.alg_var = tk.StringVar(value=self.algorithm)
        alg_menu = ttk.Combobox(
            param_frame,
            textvariable=self.alg_var,
            values=["wolff", "sw", "single"],
            width=10,
        )
        alg_menu.pack(side=tk.LEFT)

        # 更新按钮
        update_button = tk.Button(
            param_frame, text="Update Model", command=self.update_model
        )
        update_button.pack(side=tk.LEFT)

        # 绘图按钮
        plot_button = tk.Button(self.master, text="Plot Spins", command=self.plot_spins)
        plot_button.pack(side=tk.TOP)

        # 动画按钮
        animate_button = tk.Button(
            self.master, text="Start Animation", command=self.start_animation
        )
        animate_button.pack(side=tk.TOP)

    def update_model(self):
        # 更新模型参数
        self.L = self.L_var.get()
        self.T = self.T_var.get()
        self.algorithm = self.alg_var.get()

        # 重新创建模型和更新器
        self.model = HeisenbergFCC(L=self.L, T=self.T)
        self.updater = create_updater(self.model, self.algorithm)

        print(f"Model updated: L={self.L}, T={self.T}, Algorithm={self.algorithm}")

    def plot_spins(self):
        # 使用 Visualization 类的方法绘制自旋构型
        self.vis.plot_spins(self.model)

    def start_animation(self):
        # 使用 Visualization 类的方法启动动画
        ani = self.vis.animate_update(self.model, self.updater, steps=50)
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveApp(root)
    root.mainloop()
