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
        self.x_min = -20.0
        self.x_max = 20.0
        self.nx = 1000
        self.t_max = 10.0
        self.nt = 1000

        # 势阱参数
        self.well_depth = 1.0
        self.well_width = 4.0
        self.well_center = 0.0

        # 波包参数
        self.x0 = 5.0  # 初始位置
        self.k0 = -1.0  # 初始动量
        self.sigma = 1.0  # 波包宽度

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
        V[(self.x > well_left) & (self.x < well_right)] = self.params.well_depth
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
        # 记录初态
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
        self.psi_history = [self.psi.copy()]  # 记录初态
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
        self.psi_history = [self.psi.copy()]  # 记录初态

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
        """绘制静态图（初态、末态、3D演化图和热图）"""
        fig = plt.figure(figsize=(12, 12))
        grid = plt.GridSpec(
            5,
            2,
            height_ratios=[0.8, 0.8, 1, 1, 0.8],
            hspace=0.4,
            wspace=0.3,
        )

        # 第一行：初态的波函数幅值和相位
        ax1_amp = fig.add_subplot(grid[0, :])
        ax1_phase = ax1_amp.twinx()  # 创建共享x轴的次坐标轴

        initial_psi = self.solver.psi
        initial_amplitude = np.abs(initial_psi)
        initial_prob = initial_amplitude**2  # 计算概率密度
        initial_phase = np.angle(initial_psi)

        # 绘制幅值和概率密度（使用左轴）
        (line_amp,) = ax1_amp.plot(
            self.solver.x, initial_amplitude, "b-", label="幅值", linewidth=2
        )
        (line_prob,) = ax1_amp.plot(
            self.solver.x, initial_prob, "g-", label="概率密度", linewidth=2
        )

        # 绘制相位（使用右轴）
        (line_phase,) = ax1_phase.plot(
            self.solver.x,
            initial_phase,
            "r:",
            label="相位",
            linewidth=1.5,
            marker=".",
            markersize=3,
            markevery=5,
        )

        # 绘制势场（使用左轴）
        V_scaled = self.solver.V / self.solver.params.well_depth
        ax1_amp.fill_between(
            self.solver.x, V_scaled, alpha=0.2, color="gray", label="势场"
        )

        # 设置左轴（幅值和概率密度）
        ax1_amp.set_xlabel("x")
        ax1_amp.set_ylabel("幅值/概率密度", color="b")
        ax1_amp.tick_params(axis="y", labelcolor="b")
        ax1_amp.set_ylim(-0.1, 1.1)

        # 设置右轴（相位）
        ax1_phase.set_ylabel("相位 (rad)", color="r")
        ax1_phase.tick_params(axis="y", labelcolor="r")
        ax1_phase.set_ylim(-np.pi, np.pi)
        ax1_phase.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax1_phase.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

        # 合并图例
        lines1, labels1 = ax1_amp.get_legend_handles_labels()
        lines2, labels2 = ax1_phase.get_legend_handles_labels()
        ax1_amp.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

        ax1_amp.set_title("初态波函数")
        ax1_amp.grid(True, alpha=0.3)

        # 求解获取时间演化
        if self.psi_history is None:
            self.psi_history = self.solver.solve()

        # 第二行：末态的波函数幅值和相位
        ax2_amp = fig.add_subplot(grid[1, :])
        ax2_phase = ax2_amp.twinx()  # 创建共享x轴的次坐标轴

        final_psi = self.psi_history[-1]
        final_amplitude = np.abs(final_psi)
        final_prob = final_amplitude**2  # 计算概率密度
        final_phase = np.angle(final_psi)

        # 绘制幅值和概率密度（使用左轴）
        (line_amp,) = ax2_amp.plot(
            self.solver.x, final_amplitude, "b-", label="幅值", linewidth=2
        )
        (line_prob,) = ax2_amp.plot(
            self.solver.x, final_prob, "g-", label="概率密度", linewidth=2
        )

        # 绘制相位（使用右轴）
        (line_phase,) = ax2_phase.plot(
            self.solver.x,
            final_phase,
            "r:",
            label="相位",
            linewidth=1.5,
            marker=".",
            markersize=3,
            markevery=5,
        )

        # 绘制势场（使用左轴）
        ax2_amp.fill_between(
            self.solver.x, V_scaled, alpha=0.2, color="gray", label="势场"
        )

        # 设置左轴（幅值和概率密度）
        ax2_amp.set_xlabel("x")
        ax2_amp.set_ylabel("幅值/概率密度", color="b")
        ax2_amp.tick_params(axis="y", labelcolor="b")
        ax2_amp.set_ylim(-0.1, 1.1)

        # 设置右轴（相位）
        ax2_phase.set_ylabel("相位 (rad)", color="r")
        ax2_phase.tick_params(axis="y", labelcolor="r")
        ax2_phase.set_ylim(-np.pi, np.pi)
        ax2_phase.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax2_phase.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

        # 合并图例
        lines1, labels1 = ax2_amp.get_legend_handles_labels()
        lines2, labels2 = ax2_phase.get_legend_handles_labels()
        ax2_amp.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
        )

        ax2_amp.set_title("末态波函数")
        ax2_amp.grid(True, alpha=0.3)

        # 在第二行和第三行之间添加求解方法信息
        method_name = (
            "Crank-Nicolson"
            if isinstance(self.solver, CrankNicolsonSolver)
            else "显式方法"
        )
        info_text = (
            f"求解方法：{method_name}    "
            f"总时间步数：{self.solver.params.nt}    "
            f"时间步长：{self.solver.params.dt:.2e} ℏ/E₀"
        )

        # 调整y位置到0.62（第二行和第三行之间），x位置保持在0.5（居中）
        text_box = plt.figtext(
            0.5,
            0.62,
            info_text,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                alpha=0.8,
                edgecolor="lightgray",
                boxstyle="round,pad=0.5",
                linewidth=0.5,
            ),
            fontsize=9,
        )

        # 确保文本框不会被其他元素遮挡
        text_box.set_zorder(1000)
        # 第三行和第四行：3D图
        # 坐标空间3D图
        ax_3d_x = fig.add_subplot(grid[2:4, 0], projection="3d")
        X, T = np.meshgrid(self.solver.x, self.solver.t)
        prob_density = np.array([np.abs(psi) ** 2 for psi in self.psi_history])

        surf = ax_3d_x.plot_surface(
            X,
            T,
            prob_density,
            cmap=cmap,
            linewidth=0.5,
            antialiased=True,
            alpha=0.8,
            rcount=100,
            ccount=100,
        )

        ax_3d_x.set_title("坐标空间概率密度演化")
        ax_3d_x.set_xlabel("位置 x")
        ax_3d_x.set_ylabel("时间 t")
        ax_3d_x.set_zlabel("|ψ(x)|²")
        cbar = fig.colorbar(surf, ax=ax_3d_x, shrink=0.5, aspect=10)
        ax_3d_x.view_init(elev=25, azim=45)
        ax_3d_x.dist = 10

        # 动量空间3D图
        ax_3d_k = fig.add_subplot(grid[2:4, 1], projection="3d")

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

        surf_k = ax_3d_k.plot_surface(
            K,
            T,
            psi_k_history,
            cmap=cmap,
            linewidth=0.5,
            antialiased=True,
            alpha=0.8,
            rcount=100,
            ccount=100,
        )

        ax_3d_k.set_title("动量空间概率密度演化")
        ax_3d_k.set_xlabel("动量 k")
        ax_3d_k.set_ylabel("时间 t")
        ax_3d_k.set_zlabel("|ψ(k)|²")
        cbar_k = fig.colorbar(surf_k, ax=ax_3d_k, shrink=0.5, aspect=10)
        ax_3d_k.view_init(elev=25, azim=45)
        ax_3d_k.dist = 10

        # 第五行：热图
        # 坐标空间热图
        ax_heat_x = fig.add_subplot(grid[4, 0])
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
            cmap=cmap,
        )

        ax_heat_x.set_title("坐标空间概率密度演化")
        ax_heat_x.set_xlabel("位置 x")
        ax_heat_x.set_ylabel("时间 t")
        fig.colorbar(im_x, ax=ax_heat_x, label="|ψ(x)|²")

        # 动量空间热图
        ax_heat_k = fig.add_subplot(grid[4, 1])
        im_k = ax_heat_k.imshow(
            psi_k_history,
            aspect="auto",
            origin="lower",
            extent=[k.min(), k.max(), 0, self.solver.params.t_max],
            cmap=cmap,
        )

        ax_heat_k.set_title("动量空间概率密度演化")
        ax_heat_k.set_xlabel("动量 k")
        ax_heat_k.set_ylabel("时间 t")
        fig.colorbar(im_k, ax=ax_heat_k, label="|ψ(k)|²")

        # 调整整体布局
        fig.subplots_adjust(
            top=0.95,
            bottom=0.08,
            left=0.1,
            right=0.9,
            hspace=0.4,
            wspace=0.3,
        )

        return fig

    def setup_animation(self):
        """设置动画"""
        if self.psi_history is None:
            self.psi_history = self.solver.solve()

        # 初始化动画状态
        self.animation_running = True
        self.current_frame = 0  # 添加current_frame变量

        # 创建图形
        fig = plt.figure(figsize=(12, 8))

        # 创建主坐标轴和次坐标轴
        ax1 = plt.subplot(111)
        ax2 = ax1.twinx()  # 创建共享x轴的次坐标轴

        # 添加滑动条
        plt.subplots_adjust(bottom=0.15)  # 为滑动条留出空间
        ax_time = plt.axes([0.2, 0.05, 0.6, 0.03])

        # 计算准确的最大时间值
        max_time = self.solver.params.t_max
        time_slider = plt.Slider(
            ax=ax_time,
            label="时间 (ℏ/E₀)",
            valmin=0,
            valmax=max_time,
            valinit=0,
            valstep=self.solver.params.dt,
        )

        # 绘制势场
        V_scaled = self.solver.V / self.solver.params.well_depth
        ax1.fill_between(self.solver.x, V_scaled, alpha=0.2, color="gray", label="势场")

        # 初始化线条（幅值和概率密度用左轴，相位用右轴）
        (line_amp,) = ax1.plot([], [], "b-", label="幅值", linewidth=2)
        (line_prob,) = ax1.plot([], [], "g-", label="概率密度", linewidth=2)
        # 使用点线表示相位，降低密度
        (line_phase,) = ax2.plot(
            [],
            [],
            "r:",
            label="相位",
            linewidth=1.5,
            marker=".",
            markersize=3,
            markevery=5,  # 每5个点显示一个标记
        )

        # 设置坐标轴
        ax1.set_xlim(self.solver.params.x_min, self.solver.params.x_max)
        ax1.set_ylim(-0.1, 1.1)  # 调整幅值和概率密度的范围
        ax2.set_ylim(-np.pi, np.pi)  # 相位范围从-π到π

        # 设置标签
        ax1.set_xlabel("位置 x")
        ax1.set_ylabel("幅值/概率密度", color="b")
        ax2.set_ylabel("相位 (rad)", color="r")

        # 设置刻度颜色
        ax1.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")

        # 在相位轴上添加π刻度
        ax2.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax2.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

        # 添加图例，调整位置和透明度
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            framealpha=0.8,
        )

        # 添加网格
        ax1.grid(True, alpha=0.3)

        def init():
            """初始化动画"""
            line_amp.set_data([], [])
            line_phase.set_data([], [])
            line_prob.set_data([], [])
            return line_amp, line_phase, line_prob

        def update(frame):
            """更新动画帧"""
            frame = int(frame)
            if frame >= len(self.psi_history):
                return line_amp, line_phase, line_prob

            psi = self.psi_history[frame]
            amplitude = np.abs(psi)
            phase = np.angle(psi)
            prob = amplitude**2

            line_amp.set_data(self.solver.x, amplitude)
            line_phase.set_data(self.solver.x, phase)
            line_prob.set_data(self.solver.x, prob)

            # 更新标题，使用准确的时间值
            current_time = frame * self.solver.params.dt
            ax1.set_title(f"t = {current_time:.3f} ℏ/E₀")

            # 更新滑动条位置而不触发事件
            time_slider.eventson = False  # 禁用事件
            time_slider.set_val(current_time)
            time_slider.eventson = True  # 重新启用事件

            return line_amp, line_phase, line_prob

        def slider_update(val):
            """滑动条更新函数"""
            # 使用比例计算确保精确对应
            frame = int((val / max_time) * (len(self.psi_history) - 1))
            self.current_frame = frame  # 更新当前帧

            # 如果动画正在运行，暂停它
            if hasattr(fig, "_animation") and self.animation_running:
                fig._animation.event_source.stop()

            # 更新图形
            update(frame)
            fig.canvas.draw_idle()

            # 如果动画之前在运行，恢复它
            if hasattr(fig, "_animation") and self.animation_running:
                fig._animation.event_source.start()

        # 设置滑动条回调
        time_slider.on_changed(slider_update)

        # 添加暂停/继续按钮
        pause_ax = plt.axes([0.85, 0.05, 0.1, 0.03])
        pause_button = plt.Button(pause_ax, "Pause")

        # 保存控件为图形的属性，防止被垃圾回收
        fig.slider = time_slider
        fig.pause_button = pause_button

        # 创建帧生成器
        def frame_generator():
            while True:
                yield self.current_frame
                self.current_frame = (self.current_frame + 1) % len(self.psi_history)

        # 保存动画数据
        fig._animation_data = {
            "init": init,
            "update": update,
            "lines": (line_amp, line_phase, line_prob),
        }

        return fig, init, update, pause_button, frame_generator  # 返回frame_generator

    def animate(self, frame_interval=20):
        """创建和显示动画"""
        fig, init, update, pause_button, frame_generator = self.setup_animation()

        def toggle_animation(event):
            if hasattr(fig, "_animation"):
                if self.animation_running:
                    fig._animation.event_source.stop()
                    pause_button.label.set_text("Continue")
                else:
                    fig._animation.event_source.start()
                    pause_button.label.set_text("Pause")
                self.animation_running = not self.animation_running

        pause_button.on_clicked(toggle_animation)

        # 设置窗口标题
        fig.canvas.manager.set_window_title("波函数演化动画")

        # 创建动画并保存引用
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=frame_generator(),  # 使用自定义帧生成器
            interval=frame_interval,
            blit=False,
            repeat=True,
            save_count=len(self.psi_history),  # 保存所有帧
        )

        # 保存动画对象的引用到图形中
        fig._animation = anim

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

        # 添加进度条
        self.progress = ttk.Progressbar(frame, length=200, mode="determinate")
        self.progress.grid(row=2, column=0, columnspan=3, pady=5)

        # 添加状态标签
        self.status_label = ttk.Label(frame, text="就绪")
        self.status_label.grid(row=3, column=0, columnspan=3)

        def show_static():
            """显示静态图"""
            # 创建求解器
            if method_var.get() == "CN":
                self.solver = CrankNicolsonSolver(self.params)
            else:
                self.solver = ExplicitSolver(self.params)

            # 创建可视化器并显示
            self.visualizer = Visualizer(self.solver)
            fig_static = self.visualizer.plot_static()
            plt.show()

        def show_animation():
            """显示动画的函数"""
            try:
                self.status_label.config(text="正在准备动画...")
                self.progress["value"] = 0
                self.root.update()

                # 创建求解器（如果需要）
                if self.solver is None:
                    if method_var.get() == "CN":
                        self.solver = CrankNicolsonSolver(self.params)
                    else:
                        self.solver = ExplicitSolver(self.params)
                    self.visualizer = Visualizer(self.solver)

                # 计算时间演化
                self.status_label.config(text="正在计算时间演化...")
                self.root.update()

                if self.visualizer.psi_history is None:
                    total_steps = self.params.nt
                    for i, _ in enumerate(self.solver.solve()):
                        self.progress["value"] = (i + 1) / total_steps * 100
                        if i % 100 == 0:  # 每100步更新一次UI
                            self.root.update()

                self.status_label.config(text="正在生成动画...")
                self.root.update()

                # 直接创建和显示动画，不创建额外的空白图形
                anim = self.visualizer.animate()
                plt.show()

                self.status_label.config(text="动画已显示")
                self.progress["value"] = 100
                self.root.update()

            except Exception as e:
                self.status_label.config(text=f"错误: {str(e)}")
                self.progress["value"] = 0

        # 添加按钮
        ttk.Button(frame, text="显示静态图", command=show_static).grid(
            row=1, column=0, columnspan=2, pady=5
        )
        ttk.Button(frame, text="显示动画", command=show_animation).grid(
            row=1, column=2, columnspan=1, pady=5
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
