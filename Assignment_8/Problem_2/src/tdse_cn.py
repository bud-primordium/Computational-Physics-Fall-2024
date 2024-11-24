import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import platform
import logging
import gc


def configure_matplotlib_fonts():
    """配置matplotlib的字体设置"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    system = platform.system()

    if system == "Darwin":  # macOS
        plt.rcParams["font.family"] = ["Arial Unicode MS"]
    elif system == "Windows":
        plt.rcParams["font.family"] = ["Microsoft YaHei"]
    else:  # Linux
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]

    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


class SchrodingerSolver:
    """含时薛定谔方程求解器"""

    def __init__(self, x_min=-10, x_max=10, Nx=500, dt=0.01, T=5):
        # 空间离散化
        self.x = np.linspace(x_min, x_max, Nx)
        self.dx = self.x[1] - self.x[0]
        self.Nx = Nx

        # 时间离散化（考虑稳定性条件）
        stability_dt = self.dx**2 / 2  # 前向方案的稳定性条件
        self.dt = min(dt, stability_dt)  # 确保稳定性
        self.T = T
        self.Nt = int(T / self.dt)

        # 物理常数（原子单位制）
        self.hbar = 1.0
        self.m = 1.0
        self.time_unit = "ℏ/E₀"

        # 势场（方形势阱）
        self.V = np.zeros(Nx)
        self.V[(self.x < -5) | (self.x > 5)] = 10.0

        # 初始波包参数
        self.k0 = 2.0  # 初始动量
        self.x0 = -5.0  # 初始位置
        self.sigma = 1.0  # 波包宽度

    def initial_state(self):
        """生成归一化的高斯波包初态"""
        psi = np.sqrt(1 / np.pi) * np.exp(
            1j * self.k0 * self.x - (self.x - self.x0) ** 2 / (2 * self.sigma**2)
        )
        # 波函数归一化
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
        return psi / norm

    def solve(self, theta):
        """通用求解器
        Args:
            theta: 混合参数，0表示前向，1表示后向，0.5表示CN格式
        """
        psi = self.initial_state()
        alpha = 1j * self.hbar * self.dt / (2 * self.m * self.dx**2)
        results = [psi.copy()]

        # 构建矩阵
        diagonals = [
            theta * np.ones(self.Nx - 1) * alpha,
            -2 * theta * alpha * np.ones(self.Nx)
            - 1j * self.dt * self.V / self.hbar
            - 1,
            theta * np.ones(self.Nx - 1) * alpha,
        ]
        A = diags(diagonals, [-1, 0, 1], format="csc")

        B = -(1 - theta) * diags(
            [alpha, -2 * alpha, alpha], [-1, 0, 1], shape=(self.Nx, self.Nx)
        )

        for n in range(self.Nt):
            psi = spsolve(A, (B @ psi - psi))
            # 每步都重新归一化以控制数值误差
            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * self.dx)
            psi = psi / norm
            if n % 5 == 0:  # 每5步保存一次结果
                results.append(psi.copy())

        return results


class SchrodingerSolverGUI:
    """GUI界面类"""

    def __init__(self, master):
        self.master = master
        self.master.title("含时薛定谔方程求解器")

        # 配置字体
        configure_matplotlib_fonts()

        # 主框架
        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # 初始化求解器和其他变量
        self.solver = SchrodingerSolver()
        self.results = None
        self.frame_idx = 0
        self.is_playing = False

        # 创建界面元素
        self.create_control_panel()
        self.create_animation_controls()
        self.create_plot_area()

        # 绑定窗口关闭事件
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_control_panel(self):
        """创建控制面板"""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 方法选择（左侧）
        method_frame = ttk.Frame(control_frame)
        method_frame.pack(side=tk.LEFT, padx=5)

        ttk.Label(method_frame, text="方法:").pack(side=tk.LEFT)
        self.method_var = tk.StringVar(value="Crank-Nicolson (θ=1/2)")
        self.method_menu = ttk.Combobox(
            method_frame,
            textvariable=self.method_var,
            values=[
                "前向显式 (θ=0)",
                "后向隐式 (θ=1)",
                "Crank-Nicolson (θ=1/2)",
                "方法对比",
                "自定义θ",
            ],
            width=20,
        )
        self.method_menu.pack(side=tk.LEFT, padx=5)

        # θ值控制（中间）
        theta_frame = ttk.Frame(control_frame)
        theta_frame.pack(side=tk.LEFT, padx=5)

        self.theta_var = tk.DoubleVar(value=0.5)
        ttk.Label(theta_frame, text="θ:").pack(side=tk.LEFT)
        self.theta_slider = ttk.Scale(
            theta_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.theta_var,
            length=100,
        )

        self.theta_value_label = ttk.Label(
            theta_frame, text=f"{self.theta_var.get():.2f}"
        )
        self.theta_value_label.pack(side=tk.LEFT)

        # 运行按钮（右侧）
        self.run_button = ttk.Button(
            control_frame, text="开始模拟", command=self.run_simulation
        )
        self.run_button.pack(side=tk.RIGHT, padx=5)

        # 绑定事件
        self.method_menu.bind("<<ComboboxSelected>>", self.update_theta_display)
        self.theta_var.trace("w", self.format_theta)
        self.update_theta_display()

    def create_animation_controls(self):
        """创建动画控制面板"""
        anim_frame = ttk.Frame(self.main_frame)
        anim_frame.pack(fill=tk.X, padx=5, pady=5)

        # 播放控制（左侧）
        self.play_button = ttk.Button(
            anim_frame, text="▶", width=3, command=self.toggle_animation
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        # 时间显示（右侧）
        self.time_label = ttk.Label(
            anim_frame, text=f"时间: 0.00 / {self.solver.T:.2f} {self.solver.time_unit}"
        )
        self.time_label.pack(side=tk.RIGHT, padx=5)

        # 时间轴滑块（中间）
        self.time_var = tk.DoubleVar(value=0)
        self.time_slider = ttk.Scale(
            anim_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.time_var,
            command=self.update_frame,
        )
        self.time_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def create_plot_area(self):
        """创建绘图区域"""
        self.fig, self.ax = plt.subplots(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def format_theta(self, *args):
        """格式化θ值显示"""
        try:
            value = self.theta_var.get()
            formatted = f"{value:.2f}"
            self.theta_value_label.config(text=formatted)
        except:
            pass

    def update_theta_display(self, event=None):
        """更新θ值显示方式"""
        method = self.method_var.get()

        if "自定义θ" in method:
            self.theta_slider.pack(side=tk.LEFT, padx=5)
        else:
            self.theta_slider.pack_forget()

        if "前向显式" in method:
            self.theta_var.set(0.0)
        elif "后向隐式" in method:
            self.theta_var.set(1.0)
        elif "Crank-Nicolson" in method:
            self.theta_var.set(0.5)

    def update_frame(self, *args):
        """更新动画帧"""
        if self.results is not None:
            idx = int(self.time_var.get() / 100 * (len(self.results) - 1))
            self.frame_idx = idx
            self.plot_frame(idx)

            current_time = idx * self.solver.dt * 5
            self.time_label.config(
                text=f"时间: {current_time:.2f} / {self.solver.T:.2f} {self.solver.time_unit}"
            )

    def plot_frame(self, idx):
        """绘制指定帧"""
        self.ax.clear()

        if isinstance(self.results, dict):
            for name, results in self.results.items():
                if idx < len(results):
                    psi = results[idx]
                    self.ax.plot(self.solver.x, np.abs(psi) ** 2, label=name)
            self.ax.legend(fontsize="small", loc="upper right")
        else:
            psi = self.results[idx]
            self.ax.plot(self.solver.x, np.abs(psi) ** 2)

        self.ax.plot(self.solver.x, self.solver.V / 10, "k--", label="势场（缩放）")
        self.ax.set_ylim(0, 0.5)
        self.ax.set_xlabel("位置 (a₀)")
        self.ax.set_ylabel("概率密度")
        self.ax.grid(True)
        self.canvas.draw()

    def toggle_animation(self):
        """切换动画播放状态"""
        self.is_playing = not self.is_playing
        self.play_button.configure(text="⏸" if self.is_playing else "▶")
        if self.is_playing:
            self.animate()

    def animate(self):
        """执行动画更新"""
        if self.is_playing and self.results is not None:
            self.frame_idx = (self.frame_idx + 1) % len(self.results)
            self.time_var.set(self.frame_idx / (len(self.results) - 1) * 100)
            self.plot_frame(self.frame_idx)
            self.master.after(50, self.animate)

    def run_simulation(self):
        """运行模拟"""
        # 停止当前动画并清理
        if self.is_playing:
            self.toggle_animation()
        if self.results is not None:
            del self.results
            gc.collect()

        method = self.method_var.get()
        theta = self.theta_var.get()

        # 执行计算
        if "方法对比" in method:
            self.results = {
                "前向显式 (θ=0)": self.solver.solve(0.0),
                "后向隐式 (θ=1)": self.solver.solve(1.0),
                "Crank-Nicolson (θ=1/2)": self.solver.solve(0.5),
            }
            # 确保所有方法使用相同帧数
            min_length = min(len(results) for results in self.results.values())
            self.results = {
                name: results[:min_length] for name, results in self.results.items()
            }
        else:
            self.results = self.solver.solve(theta)

        # 重置动画状态
        self.frame_idx = 0
        self.time_var.set(0)
        self.plot_frame(0)
        self.is_playing = False
        self.toggle_animation()

    def on_closing(self):
        """窗口关闭时的清理工作"""
        self.is_playing = False
        plt.close(self.fig)
        if self.results is not None:
            del self.results
        gc.collect()
        self.master.destroy()


def main():
    root = tk.Tk()
    app = SchrodingerSolverGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
