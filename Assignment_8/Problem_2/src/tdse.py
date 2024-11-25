import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from typing import Tuple, Optional
import logging
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import platform

# 选择Google开发的改进版彩虹色谱
cmap = colormaps.get_cmap("turbo")


def setup_logger() -> logging.Logger:
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger("SchrodingerSolver")


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


class Parameters:
    """参数类"""

    def __init__(self):
        # 网格参数
        self.x_min = -10.0
        self.x_max = 10.0
        self.nx = 500
        self.t_max = 1.0
        self.nt = 10000

        # 势阱参数
        self.well_depth = 1.0
        self.well_width = 2.0
        self.well_center = 0.0

        # 波包参数
        self.x0 = -5.0  # 初始位置
        self.k0 = 2.0  # 初始动量
        self.sigma = 0.5  # 波包宽度

        # 计算导出参数
        self.update_derived_params()

    def update_derived_params(self):
        """更新导出参数"""
        self.dx = (self.x_max - self.x_min) / (self.nx - 1)
        self.dt = self.t_max / (self.nt - 1)


class SchrodingerSolver:
    """求解器基类"""

    def __init__(self, params: Parameters):
        self.params = params
        self.logger = setup_logger()

        # 创建空间网格
        self.x = np.linspace(params.x_min, params.x_max, params.nx)
        self.t = np.linspace(0, params.t_max, params.nt)

        # 创建势阱
        self.V = self.create_potential()

        # 初始化波函数
        self.psi = self.create_initial_state()

        # 存储演化历史
        self.psi_history = []

        self.check_stability()

    def create_potential(self) -> np.ndarray:
        """创建势阱"""
        V = np.zeros_like(self.x)
        well_left = self.params.well_center - self.params.well_width / 2
        well_right = self.params.well_center + self.params.well_width / 2
        V[self.x < well_left] = self.params.well_depth
        V[self.x > well_right] = self.params.well_depth
        return V

    def create_initial_state(self) -> np.ndarray:
        """创建初始高斯波包"""
        return np.sqrt(1 / np.pi) * np.exp(
            -((self.x - self.params.x0) ** 2) / (2 * self.params.sigma**2)
            + 1j * self.params.k0 * self.x
        )

    def check_stability(self):
        """检查稳定性条件"""
        stability_dt = self.params.dx**2 / 2
        if self.params.dt > stability_dt:
            self.logger.warning(
                f"警告: 时间步长不满足稳定性条件! "
                f"当前: {self.params.dt:.2e}, 要求: ≤{stability_dt:.2e}"
            )

    def step(self):
        """单步时间演化，需要在子类中实现"""
        raise NotImplementedError("子类必须实现step方法")

    def solve(self):
        """求解完整时间演化"""
        # 记录初始态
        self.psi_history = [self.psi.copy()]

        # 进行时间演化
        for _ in range(self.params.nt - 1):
            self.step()

        return self.psi_history


class CrankNicolsonSolver(SchrodingerSolver):
    """CN求解器"""

    def __init__(self, params: Parameters):
        super().__init__(params)
        # 预计算三对角矩阵系数
        self.alpha = 1j * self.params.dt / (4 * self.params.dx**2)

        # 构建三对角矩阵
        diagonal = 1 + 2 * self.alpha + 1j * self.params.dt * self.V / 2
        off_diagonal = -self.alpha * np.ones(self.params.nx - 1)

        # 左边矩阵 (1 + iΔtH/2)
        self.matrix_left = diags(
            [off_diagonal, diagonal, off_diagonal],
            [-1, 0, 1],
            shape=(self.params.nx, self.params.nx),
        ).tocsc()

        # 右边矩阵 (1 - iΔtH/2)
        self.matrix_right = diags(
            [
                self.alpha,
                1 - 2 * self.alpha - 1j * self.params.dt * self.V / 2,
                self.alpha,
            ],
            [-1, 0, 1],
            shape=(self.params.nx, self.params.nx),
        ).tocsc()

    def step(self):
        """CN方法时间推进一步"""
        # 计算右边 (1 - iΔtH/2)ψ
        rhs = self.matrix_right @ self.psi
        # 求解线性方程组 (1 + iΔtH/2)ψ^(n+1) = rhs
        self.psi = spsolve(self.matrix_left, rhs)
        # 记录历史
        self.psi_history.append(self.psi.copy())

    def solve(self):
        """求解完整时间演化"""
        self.psi_history = [self.psi.copy()]  # 记录初始态
        for _ in range(self.params.nt - 1):
            self.step()
        return self.psi_history


class ExplicitSolver(SchrodingerSolver):
    """显式求解器"""

    def __init__(self, params: Parameters):
        super().__init__(params)
        self.psi_prev = self.psi.copy()  # 存储前一步的波函数 ψ^(n-1)

        # 计算拉普拉斯算子系数 iΔt/Δx²
        self.alpha = 1j * self.params.dt / self.params.dx**2

    def step(self):
        """显式方法时间推进一步
        实现公式：ψ^(n+1) = ψ^(n-1) + iΔt/Δx²(ψ^n_(j+1) + ψ^n_(j-1) - 2ψ^n_j) - 2iΔtV_jψ^n_j
        """
        # 保存当前步的波函数 ψ^n 用于计算
        psi_current = self.psi.copy()

        # 计算 ψ^n_(j+1)：将数组向右移动一位
        # 例如[1,2,3,4,5] -> [5,1,2,3,4]，实现了周期边界条件
        psi_jp1 = np.roll(psi_current, 1)

        # 计算 ψ^n_(j-1)：将数组向左移动一位
        psi_jm1 = np.roll(psi_current, -1)

        # 计算拉普拉斯项 (ψ^n_(j+1) + ψ^n_(j-1) - 2ψ^n_j)
        laplacian = psi_jp1 + psi_jm1 - 2 * psi_current

        # 计算新的波函数 ψ^(n+1)
        self.psi = (
            self.psi_prev
            + self.alpha * laplacian
            - 2j * self.params.dt * self.V * psi_current
        )

        # 更新前一步的波函数，为下一步计算做准备
        self.psi_prev = psi_current

        # 记录演化历史
        self.psi_history.append(self.psi.copy())

    def solve(self):
        """求解完整时间演化"""
        self.psi_history = [self.psi.copy()]  # 记录初始态

        # 使用CN方法计算第一步
        cn_first_step = CrankNicolsonSolver(self.params)
        cn_first_step.psi = self.psi.copy()
        cn_first_step.step()
        self.psi = cn_first_step.psi
        self.psi_history.append(self.psi.copy())

        # 继续使用显式方法
        for _ in range(self.params.nt - 2):
            self.step()

        return self.psi_history


class Visualizer:
    """可视化类"""

    def __init__(self, solver: SchrodingerSolver):
        self.solver = solver
        self.psi_history = None
        configure_matplotlib_fonts()

    def plot_static(self):
        """绘制静态图（波函数幅值相位、3D演化图和热图）"""
        fig = plt.figure(figsize=(15, 20))
        grid = plt.GridSpec(4, 2, height_ratios=[1, 2, 2, 1.2])

        # 第一行：波函数幅值和相位
        ax1 = fig.add_subplot(grid[0, :])
        psi = self.solver.psi
        amplitude = np.abs(psi)
        phase = np.angle(psi)

        ax1.plot(self.solver.x, amplitude, label="幅值", color="blue", linewidth=2)
        ax1.plot(self.solver.x, phase, label="相位", color="red", linewidth=2)

        # 绘制势场
        V_scaled = self.solver.V / self.solver.params.well_depth
        ax1.fill_between(self.solver.x, V_scaled, alpha=0.2, color="gray", label="势场")

        ax1.set_title("波函数幅值和相位")
        ax1.set_xlabel("x")
        ax1.set_ylabel("幅值/相位")
        ax1.legend()
        ax1.grid(True)

        # 求解获取时间演化
        if self.psi_history is None:
            self.psi_history = self.solver.solve()

        # 第二行：3D图
        # 坐标空间3D图
        ax_3d_x = fig.add_subplot(grid[1:3, 0], projection="3d")
        X, T = np.meshgrid(self.solver.x, self.solver.t)
        prob_density = np.array([np.abs(psi) ** 2 for psi in self.psi_history])

        # 使用surface plot with colormap
        surf = ax_3d_x.plot_surface(
            X,
            T,
            prob_density,
            cmap="turbo",
            linewidth=0.5,
            antialiased=True,
            alpha=0.8,
            rcount=100,  # 调整网格密度
            ccount=100,
        )  # 调整网格密度

        ax_3d_x.set_title("坐标空间概率密度演化")
        ax_3d_x.set_xlabel("位置 x")
        ax_3d_x.set_ylabel("时间 t")
        ax_3d_x.set_zlabel("|ψ(x)|²")
        # 调整颜色条位置和大小
        cbar = fig.colorbar(surf, ax=ax_3d_x, shrink=0.5, aspect=10)
        ax_3d_x.view_init(elev=20, azim=45)  # 调整视角
        ax_3d_x.dist = 11  # 调整观察距离

        # 动量空间3D图
        ax_3d_k = fig.add_subplot(grid[1:3, 1], projection="3d")

        # 计算动量空间波函数
        k = np.fft.fftfreq(self.solver.params.nx, d=self.solver.params.dx) * 2 * np.pi
        k = np.fft.fftshift(k)
        K, T = np.meshgrid(k, self.solver.t)
        psi_k_history = []

        for psi_x in self.psi_history:
            psi_k = np.fft.fft(psi_x) * self.solver.params.dx / np.sqrt(2 * np.pi)
            psi_k = np.fft.fftshift(psi_k)
            psi_k_history.append(np.abs(psi_k) ** 2)

        psi_k_history = np.array(psi_k_history)

        # 绘制动量空间3D图
        surf_k = ax_3d_k.plot_surface(
            K,
            T,
            psi_k_history,
            cmap="turbo",
            linewidth=0.5,
            antialiased=True,
            alpha=0.8,
            rcount=100,  # 调整网格密度
            ccount=100,
        )  # 调整网格密度

        ax_3d_k.set_title("动量空间概率密度演化")
        ax_3d_k.set_xlabel("动量 k")
        ax_3d_k.set_ylabel("时间 t")
        ax_3d_k.set_zlabel("|ψ(k)|²")
        # 调整颜色条位置和大小
        cbar_k = fig.colorbar(surf_k, ax=ax_3d_k, shrink=0.5, aspect=10)
        ax_3d_k.view_init(elev=20, azim=45)  # 调整视角
        ax_3d_k.dist = 11  # 调整观察距离

        # 第四行：热图
        # 坐标空间热图
        ax_heat_x = fig.add_subplot(grid[3, 0])
        im_x = ax_heat_x.imshow(
            prob_density,
            aspect="auto",
            origin="lower",
            extent=[
                self.solver.params.x_min,
                self.solver.params.x_max,
                0,
                self.solver.params.t_max,
            ],
            cmap="turbo",
        )

        ax_heat_x.set_title("坐标空间概率密度演化")
        ax_heat_x.set_xlabel("位置 x")
        ax_heat_x.set_ylabel("时间 t")
        fig.colorbar(im_x, ax=ax_heat_x, label="|ψ(x)|²")

        # 动量空间热图
        ax_heat_k = fig.add_subplot(grid[3, 1])
        im_k = ax_heat_k.imshow(
            psi_k_history,
            aspect="auto",
            origin="lower",
            extent=[k.min(), k.max(), 0, self.solver.params.t_max],
            cmap="turbo",
        )

        ax_heat_k.set_title("动量空间概率密度演化")
        ax_heat_k.set_xlabel("动量 k")
        ax_heat_k.set_ylabel("时间 t")
        fig.colorbar(im_k, ax=ax_heat_k, label="|ψ(k)|²")

        plt.tight_layout()
        return fig

    def setup_animation(self):
        """设置动画"""
        if self.psi_history is None:
            self.psi_history = self.solver.solve()

        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制势场
        V_scaled = self.solver.V / self.solver.params.well_depth
        ax.fill_between(self.solver.x, V_scaled, alpha=0.2, color="gray")

        # 初始化线条
        (line_real,) = ax.plot([], [], "b-", label="实部")
        (line_imag,) = ax.plot([], [], "r-", label="虚部")
        (line_prob,) = ax.plot([], [], "g-", label="概率密度")

        ax.set_xlim(self.solver.params.x_min, self.solver.params.x_max)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("波函数")
        ax.legend()
        ax.grid(True)

        def init():
            """初始化动画"""
            line_real.set_data([], [])
            line_imag.set_data([], [])
            line_prob.set_data([], [])
            return line_real, line_imag, line_prob

        def animate(frame):
            """更新动画帧"""
            psi = self.psi_history[frame]
            line_real.set_data(self.solver.x, np.real(psi))
            line_imag.set_data(self.solver.x, np.imag(psi))
            line_prob.set_data(self.solver.x, np.abs(psi) ** 2)
            ax.set_title(f"t = {frame * self.solver.params.dt:.2f}")
            return line_real, line_imag, line_prob

        return fig, init, animate

    def animate(self):
        """创建和显示动画"""
        fig, init, animate = self.setup_animation()
        anim = FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(self.psi_history),
            interval=50,  # 每帧间隔50ms
            blit=True,
        )
        plt.close(fig)  # 防止显示静态图
        return anim


class SchrodingerGUI:
    """GUI界面"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("含时Schrödinger方程求解器")

        # 创建参数和求解器实例
        self.params = Parameters()
        self.solver = None
        self.visualizer = None

        # 设置GUI组件
        self.setup_gui()

    def setup_gui(self):
        """设置GUI组件"""
        # 势阱参数框架
        well_frame = ttk.LabelFrame(self.root, text="势阱参数")
        well_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.setup_well_controls(well_frame)

        # 波包参数框架
        wave_frame = ttk.LabelFrame(self.root, text="波包参数")
        wave_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.setup_wave_controls(wave_frame)

        # 求解控制框架
        ctrl_frame = ttk.LabelFrame(self.root, text="求解控制")
        ctrl_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        self.setup_solver_controls(ctrl_frame)

    def setup_well_controls(self, frame):
        """设置势阱控制组件"""
        # 势阱深度
        ttk.Label(frame, text="势阱深度:").grid(row=0, column=0, padx=5, pady=2)
        depth_var = tk.DoubleVar(value=self.params.well_depth)
        depth_entry = ttk.Entry(frame, textvariable=depth_var, width=10)
        depth_entry.grid(row=0, column=1, padx=5, pady=2)

        # 势阱宽度
        ttk.Label(frame, text="势阱宽度:").grid(row=1, column=0, padx=5, pady=2)
        width_var = tk.DoubleVar(value=self.params.well_width)
        width_entry = ttk.Entry(frame, textvariable=width_var, width=10)
        width_entry.grid(row=1, column=1, padx=5, pady=2)

        # 势阱中心位置
        ttk.Label(frame, text="中心位置:").grid(row=2, column=0, padx=5, pady=2)
        center_var = tk.DoubleVar(value=self.params.well_center)
        center_entry = ttk.Entry(frame, textvariable=center_var, width=10)
        center_entry.grid(row=2, column=1, padx=5, pady=2)

        def update_well_params():
            """更新势阱参数"""
            self.params.well_depth = depth_var.get()
            self.params.well_width = width_var.get()
            self.params.well_center = center_var.get()
            if self.solver:
                self.solver.V = self.solver.create_potential()

        ttk.Button(frame, text="更新势阱", command=update_well_params).grid(
            row=3, column=0, columnspan=2, pady=5
        )

    def setup_wave_controls(self, frame):
        """设置波包控制组件"""
        # 初始位置
        ttk.Label(frame, text="初始位置:").grid(row=0, column=0, padx=5, pady=2)
        x0_var = tk.DoubleVar(value=self.params.x0)
        x0_entry = ttk.Entry(frame, textvariable=x0_var, width=10)
        x0_entry.grid(row=0, column=1, padx=5, pady=2)

        # 初始动量
        ttk.Label(frame, text="初始动量:").grid(row=1, column=0, padx=5, pady=2)
        k0_var = tk.DoubleVar(value=self.params.k0)
        k0_entry = ttk.Entry(frame, textvariable=k0_var, width=10)
        k0_entry.grid(row=1, column=1, padx=5, pady=2)

        # 波包宽度
        ttk.Label(frame, text="波包宽度:").grid(row=2, column=0, padx=5, pady=2)
        sigma_var = tk.DoubleVar(value=self.params.sigma)
        sigma_entry = ttk.Entry(frame, textvariable=sigma_var, width=10)
        sigma_entry.grid(row=2, column=1, padx=5, pady=2)

        def update_wave_params():
            """更新波包参数"""
            self.params.x0 = x0_var.get()
            self.params.k0 = k0_var.get()
            self.params.sigma = sigma_var.get()
            if self.solver:
                self.solver.psi = self.solver.create_initial_state()

        ttk.Button(frame, text="更新波包", command=update_wave_params).grid(
            row=3, column=0, columnspan=2, pady=5
        )

    def setup_solver_controls(self, frame):
        """设置求解器控制组件"""
        # 求解方法选择
        ttk.Label(frame, text="求解方法:").grid(row=0, column=0, padx=5, pady=2)
        method_var = tk.StringVar(value="CN")
        ttk.Radiobutton(
            frame, text="Crank-Nicolson", variable=method_var, value="CN"
        ).grid(row=0, column=1)
        ttk.Radiobutton(
            frame, text="显式方法", variable=method_var, value="Explicit"
        ).grid(row=0, column=2)

        def start_solve():
            """开始求解"""
            # 创建求解器
            if method_var.get() == "CN":
                self.solver = CrankNicolsonSolver(self.params)
            else:
                self.solver = ExplicitSolver(self.params)

            # 创建可视化器并显示
            self.visualizer = Visualizer(self.solver)
            fig = self.visualizer.plot_static()
            plt.show()

        ttk.Button(frame, text="开始求解", command=start_solve).grid(
            row=1, column=0, columnspan=3, pady=5
        )

    def run(self):
        """运行GUI"""
        self.root.mainloop()


def main():
    """主程序"""
    gui = SchrodingerGUI()
    gui.run()


if __name__ == "__main__":
    main()
