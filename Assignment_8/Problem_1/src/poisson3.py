import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import logging
from typing import Tuple, List, Optional
from matplotlib.animation import FuncAnimation


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

    # 备用字体
    plt.rcParams["font.sans-serif"] = [
        "Arial Unicode MS",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def analytic_solution_case_a(x, y, Lx, Ly, V0, tol=1e-10, min_n=50, max_n=200):
    """
    计算案例(a)的解析解，动态决定需要的傅里叶项数以满足指定的容差，同时确保至少使用min_n项。

    参数:
        x, y: 网格坐标（NumPy数组）
        Lx, Ly: 区域长度和宽度 (m)
        V0: 顶部边界电势 (V)
        tol: 解析解的容差
        min_n: 最小傅里叶项的n值
        max_n: 最大傅里叶项的n值，防止无限循环

    返回:
        phi: 解析解的电势矩阵
        n_used: 实际使用的最大n值
    """
    phi = np.zeros_like(x)
    n_used = 0  # 实际使用的最大n值

    for n in range(1, max_n + 1, 2):  # 仅奇数项
        term = (
            4
            * V0
            / (n * np.pi)
            * np.sin(n * np.pi * x / Lx)
            * np.sinh(n * np.pi * y / Lx)
            / np.sinh(n * np.pi * Ly / Lx)
        )
        phi += term
        n_used = n  # 更新为当前的n值

        # 检查当前项的最大贡献是否小于容差，并且已经达到最小n值
        if n >= min_n and np.max(np.abs(term)) < tol:
            print(f"解析解在n={n}项时达到收敛，最大项贡献为{np.max(np.abs(term)):.2e}")
            break
    else:
        print(
            f"解析解未在最大n值{max_n}内收敛，最后一项贡献为{np.max(np.abs(term)):.2e}"
        )

    return phi, n_used


class PoissonSolver:
    def __init__(
        self,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        rho: np.ndarray,
        boundary_values: dict,
        epsilon0: float = 8.854e-12,
    ):
        """
        初始化求解器
        参数:
            Lx, Ly: 计算区域的长度和宽度 (m)
            Nx, Ny: x和y方向的网格点数
            rho: 电荷密度矩阵 (C/m³)
            boundary_values: 边界条件字典
            epsilon0: 真空介电常数 (F/m)
        """
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.rho = rho
        self.boundary_values = boundary_values
        self.epsilon0 = epsilon0

        # 计算网格间距
        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)

        # 初始化电势矩阵
        self.phi = np.zeros((Nx, Ny))
        self.set_boundary_conditions()

        self.phi_history = []  # 存储迭代历史，包含迭代次数和phi矩阵

    def set_boundary_conditions(self):
        """设置边界条件"""
        self.phi[:, 0] = self.boundary_values["bottom"]  # 底边界
        self.phi[:, -1] = self.boundary_values["top"]  # 顶边界
        self.phi[0, :] = self.boundary_values["left"]  # 左边界
        self.phi[-1, :] = self.boundary_values["right"]  # 右边界

    def get_optimal_omega(self) -> float:
        """计算SOR方法的最优松弛因子"""
        N = max(self.Nx, self.Ny)
        return 2 / (1 + np.sin(np.pi / N))

    def solve(
        self,
        method: str = "sor",
        tolerance: float = 1e-8,
        max_iter: int = 10000,
        save_history: bool = False,
        history_interval: int = None,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """求解泊松方程"""
        self.phi_history = []  # 清空历史记录

        # 根据方法预估收敛所需迭代次数
        if method == "jacobi":
            expected_iter = int(-np.log(tolerance) * max(self.Nx, self.Ny) ** 2)
        elif method == "gauss_seidel":
            expected_iter = int(-0.5 * np.log(tolerance) * max(self.Nx, self.Ny) ** 2)
        else:  # sor
            expected_iter = int(-np.log(tolerance) * max(self.Nx, self.Ny))

        # 动态设置历史记录间隔
        if history_interval is None:
            history_interval = max(1, expected_iter // 100)

        if method == "jacobi":
            return self._solve_jacobi(
                tolerance, max_iter, save_history, history_interval
            )
        elif method == "gauss_seidel":
            return self._solve_gauss_seidel(
                tolerance, max_iter, save_history, history_interval
            )
        elif method == "sor":
            omega = self.get_optimal_omega()
            print(f"使用最优松弛因子: ω = {omega:.3f}")
            return self._solve_sor(
                omega, tolerance, max_iter, save_history, history_interval
            )
        else:
            raise ValueError(f"未知的求解方法: {method}")

    def _solve_jacobi(
        self, tolerance: float, max_iter: int, save_history: bool, history_interval: int
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """Jacobi迭代法"""
        phi_new = np.copy(self.phi)
        history = []
        start_time = time.time()

        for it in range(max_iter):
            phi_new[1:-1, 1:-1] = 0.25 * (
                self.phi[2:, 1:-1]
                + self.phi[:-2, 1:-1]
                + self.phi[1:-1, 2:]
                + self.phi[1:-1, :-2]
                + self.dx * self.dy * self.rho[1:-1, 1:-1] / self.epsilon0
            )

            max_change = np.max(np.abs(phi_new - self.phi))
            history.append(max_change)

            if save_history and it % history_interval == 0:
                # 保存(迭代次数, phi矩阵)
                self.phi_history.append((it, np.copy(self.phi)))

            self.phi, phi_new = phi_new.copy(), self.phi

            if max_change < tolerance:
                print(f"Jacobi方法在第{it+1}次迭代收敛")
                # 确保最后一帧也被保存
                if save_history and it % history_interval != 0:
                    self.phi_history.append((it, np.copy(self.phi)))
                break
        else:
            print(f"Jacobi方法达到最大迭代次数{max_iter}仍未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def _solve_gauss_seidel(
        self, tolerance: float, max_iter: int, save_history: bool, history_interval: int
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """Gauss-Seidel迭代法"""
        history = []
        start_time = time.time()

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    old_value = self.phi[i, j]
                    self.phi[i, j] = 0.25 * (
                        self.phi[i + 1, j]
                        + self.phi[i - 1, j]
                        + self.phi[i, j + 1]
                        + self.phi[i, j - 1]
                        + self.dx * self.dy * self.rho[i, j] / self.epsilon0
                    )
                    max_change = max(max_change, abs(self.phi[i, j] - old_value))

            history.append(max_change)

            if save_history and it % history_interval == 0:
                # 保存(迭代次数, phi矩阵)
                self.phi_history.append((it, np.copy(self.phi)))

            if max_change < tolerance:
                print(f"Gauss-Seidel方法在第{it+1}次迭代收敛")
                # 确保最后一帧也被保存
                if save_history and it % history_interval != 0:
                    self.phi_history.append((it, np.copy(self.phi)))
                break
        else:
            print(f"Gauss-Seidel方法达到最大迭代次数{max_iter}仍未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def _solve_sor(
        self,
        omega: float,
        tolerance: float,
        max_iter: int,
        save_history: bool,
        history_interval: int,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """SOR迭代法"""
        history = []
        start_time = time.time()

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    old_value = self.phi[i, j]
                    self.phi[i, j] = (1 - omega) * old_value + omega * 0.25 * (
                        self.phi[i + 1, j]
                        + self.phi[i - 1, j]
                        + self.phi[i, j + 1]
                        + self.phi[i, j - 1]
                        + self.dx * self.dy * self.rho[i, j] / self.epsilon0
                    )
                    max_change = max(max_change, abs(self.phi[i, j] - old_value))

            history.append(max_change)

            if save_history and it % history_interval == 0:
                # 保存(迭代次数, phi矩阵)
                self.phi_history.append((it, np.copy(self.phi)))

            if max_change < tolerance:
                print(f"SOR方法在第{it+1}次迭代收敛")
                # 确保最后一帧也被保存
                if save_history and it % history_interval != 0:
                    self.phi_history.append((it, np.copy(self.phi)))
                break
        else:
            print(f"SOR方法达到最大迭代次数{max_iter}仍未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time


def animate_solution(
    X: np.ndarray,
    Y: np.ndarray,
    phi_history: List[Tuple[int, np.ndarray]],
    method_name: str,
    problem_params: dict,
):
    """创建求解过程动画"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 从 phi_history 中提取所有的 phi 矩阵和对应的迭代次数
    iteration_numbers = [item[0] for item in phi_history]
    phi_matrices = [item[1] for item in phi_history]

    # 计算等值线的范围
    vmin = np.min([np.min(phi) for phi in phi_matrices])
    vmax = np.max([np.max(phi) for phi in phi_matrices])
    levels = np.linspace(vmin, vmax, 50)

    # 初始化等值线图
    contour = ax.contourf(X, Y, phi_matrices[0], levels=levels, cmap="viridis")

    plt.colorbar(contour, ax=ax, label="电势 (V)")

    # 启用坐标显示
    ax.format_coord = lambda x, y: f"(x,y)=({x:.3f},{y:.3f})"

    # 计算坐标轴的填充范围
    padding_x = problem_params["Lx"] * 0.1
    padding_y = problem_params["Ly"] * 0.1

    # 抽取关键帧显示动画
    n_frames = min(100, len(phi_matrices))  # 最多显示100帧
    if len(phi_matrices) > n_frames:
        frame_indices = np.linspace(0, len(phi_matrices) - 1, n_frames, dtype=int)
        phi_matrices_sampled = [phi_matrices[i] for i in frame_indices]
        iteration_numbers_sampled = [iteration_numbers[i] for i in frame_indices]
    else:
        phi_matrices_sampled = phi_matrices
        iteration_numbers_sampled = iteration_numbers

    # 定义更新函数，用于动画
    def update(frame_num):
        ax.clear()
        contour = ax.contourf(
            X,
            Y,
            phi_matrices_sampled[frame_num],
            levels=levels,
            cmap="viridis",  # 移除了.T
        )

        # 重新设置坐标轴范围
        ax.set_xlim(-padding_x, problem_params["Lx"] + padding_x)
        ax.set_ylim(-padding_y, problem_params["Ly"] + padding_y)

        # 添加边界标注
        ax.text(
            -padding_x * 0.8,
            problem_params["Ly"] / 2,
            f"φ = {problem_params['boundary_values']['left']}V",
            rotation=90,
            verticalalignment="center",
        )
        ax.text(
            problem_params["Lx"] + padding_x * 0.4,
            problem_params["Ly"] / 2,
            f"φ = {problem_params['boundary_values']['right']}V",
            rotation=-90,
            verticalalignment="center",
        )
        ax.text(
            problem_params["Lx"] / 2,
            -padding_y * 0.8,
            f"φ = {problem_params['boundary_values']['bottom']}V",
            horizontalalignment="center",
        )
        ax.text(
            problem_params["Lx"] / 2,
            problem_params["Ly"] + padding_y * 0.4,
            f"φ = {problem_params['boundary_values']['top']}V",
            horizontalalignment="center",
        )

        # 添加边界线
        ax.plot([0, problem_params["Lx"]], [0, 0], "k--", alpha=0.5)
        ax.plot(
            [0, problem_params["Lx"]],
            [problem_params["Ly"], problem_params["Ly"]],
            "k--",
            alpha=0.5,
        )
        ax.plot([0, 0], [0, problem_params["Ly"]], "k--", alpha=0.5)
        ax.plot(
            [problem_params["Lx"], problem_params["Lx"]],
            [0, problem_params["Ly"]],
            "k--",
            alpha=0.5,
        )

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

        # 启用坐标显示
        ax.format_coord = lambda x, y: f"(x,y)=({x:.3f},{y:.3f})"

        # 显示真实的迭代次数
        iteration = iteration_numbers_sampled[frame_num]
        ax.set_title(f"{method_name}方法迭代过程 - 第{iteration}次迭代")

        return [contour]

    # 创建动画
    anim = FuncAnimation(
        fig, update, frames=len(phi_matrices_sampled), interval=50, repeat=False
    )

    plt.tight_layout()
    plt.show()


def plot_final_results(
    X: np.ndarray,
    Y: np.ndarray,
    results: dict,
    phi_analytic: Optional[np.ndarray],
    case_name: str,
    problem_params: dict,
    n_used: Optional[int] = None,
):
    """绘制最终结果对比图及差异分布图"""
    num_methods = len(results)
    fig, axs = plt.subplots(num_methods + 1, 2, figsize=(15, 5 * (num_methods + 1)))

    def setup_subplot(ax, phi, title, cmap="viridis", is_diff=False):
        mesh = ax.pcolormesh(X, Y, phi, cmap=cmap, shading="gouraud")
        plt.colorbar(mesh, ax=ax, label="差异 (V)" if is_diff else "电势 (V)")

        # 自定义鼠标悬停显示格式
        def format_coord(x, y):
            i = np.searchsorted(X[:, 0], x) - 1
            j = np.searchsorted(Y[0, :], y) - 1
            if 0 <= i < X.shape[0] and 0 <= j < Y.shape[1]:
                z = phi[i, j]
                if is_diff:
                    return f"x={x:.3f}, y={y:.3f}, Δφ={z:.3f}V"
                return f"x={x:.3f}, y={y:.3f}, φ={z:.3f}V"
            return f"x={x:.3f}, y={y:.3f}"

        ax.format_coord = format_coord

        # 添加带阴影背景的边界标注
        if not is_diff:  # 只在非差异图中添加边界标注
            ax.annotate(
                f'φ = {problem_params["boundary_values"]["left"]}V',
                xy=(0, 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(-5, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=90,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

            ax.annotate(
                f'φ = {problem_params["boundary_values"]["right"]}V',
                xy=(problem_params["Lx"], 0.5),
                xycoords=("data", "axes fraction"),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                rotation=-90,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

            ax.annotate(
                f'φ = {problem_params["boundary_values"]["bottom"]}V',
                xy=(problem_params["Lx"] / 2, 0),
                xytext=(0, -5),
                textcoords="offset points",
                ha="center",
                va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

            ax.annotate(
                f'φ = {problem_params["boundary_values"]["top"]}V',
                xy=(problem_params["Lx"] / 2, problem_params["Ly"]),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        # 添加带阴影的边界线
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=problem_params["Ly"], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=problem_params["Lx"], color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title, pad=15)

    # 绘制解析解（如果有）
    if phi_analytic is not None:
        ax = axs[0, 0]
        setup_subplot(ax, phi_analytic, "解析解")

        # 在右侧子图添加注释
        ax_annotation = axs[0, 1]
        ax_annotation.axis("off")

        if n_used is not None:
            formula = r"$\phi(x,y) = \frac{4V_0}{\pi} \sum_{n=1,3,5,\dots}^{N} \frac{1}{n} \sin\left(\frac{n\pi x}{L_x}\right) \frac{\sinh\left(\frac{n\pi y}{L_x}\right)}{\sinh\left(\frac{n\pi L_y}{L_x}\right)}$"
            annotation_text = (
                f"解析解公式:\n{formula}\n\n使用的傅里叶级数中最大n: {n_used}"
            )
        else:
            annotation_text = "解析解公式未提供\n傅里叶项数未知"

        ax_annotation.text(
            0.5,
            0.5,
            annotation_text,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
    else:
        axs[0, 0].text(0.5, 0.5, "解析解未实现", ha="center", va="center", fontsize=16)
        axs[0, 0].set_title("解析解")
        axs[0, 1].axis("off")

    # 绘制数值解及差异图
    methods = ["jacobi", "gauss_seidel", "sor"]
    for idx, method in enumerate(methods, start=1):
        # 数值解
        ax = axs[idx, 0]
        result = results[method]
        title = f"{method.upper()}方法\n"
        if result["error"] is not None:
            title += f"相对误差: {result['error']:.2e}\n"
        title += f"迭代次数: {result['iterations']}"
        setup_subplot(ax, result["phi"], title)

        # 差异分布图
        ax_diff = axs[idx, 1]
        if phi_analytic is not None:
            diff = result["phi"] - phi_analytic
            setup_subplot(
                ax_diff,
                diff,
                f"{method.upper()}方法与解析解差异分布",
                cmap="seismic",
                is_diff=True,
            )
        else:
            ax_diff.text(0.5, 0.5, "无解析解", ha="center", va="center", fontsize=16)
            ax_diff.set_title(f"{method.upper()}方法与解析解差异分布")
            ax_diff.set_xlabel("x (m)")
            ax_diff.set_ylabel("y (m)")

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 添加总标题
    source_text = (
        "无源项"
        if abs(problem_params["rho_epsilon0"]) < 1e-10
        else f"均匀源项 (ρ/ε₀ = {problem_params['rho_epsilon0']} V/m²)"
    )
    fig.suptitle(f"{case_name}\n{source_text}", fontsize=16, y=0.98)

    plt.show()


def get_user_case() -> Tuple[dict, str]:
    """获取用户选择的求解案例"""
    print("\n=== 泊松方程求解器 ===")
    print("请选择要求解的案例：")
    print("a - 无源项，顶部电势为1V其余为0V")
    print("b - 均匀源项(ρ/ε₀ = 1 V/m²)，边界全为0V")
    print("c - 自定义均匀源项和边界条件")

    while True:
        case = input("\n请输入选项 (a/b/c): ").strip().lower()
        if case in ["a", "b", "c"]:
            break
        print("无效输入，请重试！")

    if case == "a":
        problem_params = {
            "Lx": 1.0,
            "Ly": 1.5,
            "rho_epsilon0": 0.0,
            "boundary_values": {"left": 0, "right": 0, "bottom": 0, "top": 1},
        }
        return problem_params, "案例(a)"

    elif case == "b":
        problem_params = {
            "Lx": 1.0,
            "Ly": 1.0,
            "rho_epsilon0": 1.0,
            "boundary_values": {"left": 0, "right": 0, "bottom": 0, "top": 0},
        }
        return problem_params, "案例(b)"

    else:  # case == 'c'
        print("\n=== 请输入自定义参数 ===")
        print("（注：所有长度单位为米(m)，电势单位为伏特(V)）")

        # 获取区域尺寸
        while True:
            try:
                Lx = float(input("请输入x方向长度: "))
                Ly = float(input("请输入y方向长度: "))
                if Lx > 0 and Ly > 0:
                    break
                print("长度必须为正数！")
            except ValueError:
                print("请输入有效的数字！")

        # 获取源项
        while True:
            try:
                rho_epsilon0 = float(input("请输入均匀源项大小(ρ/ε₀): "))
                break
            except ValueError:
                print("请输入有效的数字！")

        # 获取边界条件
        print("\n请输入边界电势值：")
        boundary_values = {}
        for boundary in ["left", "right", "bottom", "top"]:
            while True:
                try:
                    value = float(input(f"{boundary}边界电势: "))
                    boundary_values[boundary] = value
                    break
                except ValueError:
                    print("请输入有效的数字！")

        problem_params = {
            "Lx": Lx,
            "Ly": Ly,
            "rho_epsilon0": rho_epsilon0,
            "boundary_values": boundary_values,
        }
        return problem_params, "自定义案例"


def solve_and_visualize(problem_params: dict, case_name: str):
    """求解并可视化结果"""
    # 设置网格
    Nx, Ny = 50, 50
    x = np.linspace(0, problem_params["Lx"], Nx)
    y = np.linspace(0, problem_params["Ly"], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 计算解析解（仅针对案例a）
    if (
        problem_params["rho_epsilon0"] == 0
        and problem_params["boundary_values"]["top"] != 0
    ):
        V0 = problem_params["boundary_values"]["top"]
        # 计算允许的最大n值
        max_allowed_n = int(700 * problem_params["Lx"] / (np.pi * problem_params["Ly"]))
        # 设置解析解的容差不大于数值解的容差，并至少使用min_n=50项
        phi_analytic, n_used = analytic_solution_case_a(
            X,
            Y,
            problem_params["Lx"],
            problem_params["Ly"],
            V0,
            tol=1e-30,
            min_n=50,
            max_n=max_allowed_n,
        )
        print(max_allowed_n)
        print(f"解析解使用的傅里叶级数中最大n: {n_used}")
    else:
        phi_analytic = None
        n_used = None
        print("此案例无解析解")

    # 数值求解
    methods = ["jacobi", "gauss_seidel", "sor"]
    results = {}

    for method in methods:
        print(f"\n使用{method}方法求解...")
        solver = PoissonSolver(
            problem_params["Lx"],
            problem_params["Ly"],
            Nx,
            Ny,
            problem_params["rho_epsilon0"] * np.ones((Nx, Ny)),
            problem_params["boundary_values"],
        )

        phi, history, iterations, elapsed = solver.solve(
            method=method,
            save_history=True,
            history_interval=(
                max(1, iterations // 100) if "iterations" in locals() else 10
            ),
        )

        if phi_analytic is not None:
            error = np.max(np.abs(phi - phi_analytic)) / np.max(np.abs(phi_analytic))
        else:
            error = None

        results[method] = {
            "phi": phi,
            "history": history,
            "iterations": iterations,
            "elapsed": elapsed,
            "error": error,
            "phi_history": solver.phi_history,
        }

        print(f"迭代次数: {iterations}")
        print(f"求解时间: {elapsed:.4f}秒")
        if error is not None:
            print(f"相对误差: {error:.2e}")

        # 动画展示
        animate_solution(X, Y, solver.phi_history, method.upper(), problem_params)

    # 绘制最终结果对比图
    plot_final_results(X, Y, results, phi_analytic, case_name, problem_params, n_used)


def main():
    """主程序"""
    # 配置matplotlib中文字体
    configure_matplotlib_fonts()

    # 获取用户输入并求解
    problem_params, case_name = get_user_case()
    solve_and_visualize(problem_params, case_name)


if __name__ == "__main__":
    main()
