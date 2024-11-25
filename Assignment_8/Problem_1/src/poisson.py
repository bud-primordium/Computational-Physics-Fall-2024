import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import logging
from typing import Tuple, List, Optional
from matplotlib.animation import FuncAnimation
from matplotlib import colormaps

# 选择Google开发的改进版彩虹色谱
cmap = colormaps.get_cmap("turbo")


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


def analytic_solution_case_a(x, y, Lx, Ly, V0, tol=1e-20, min_n=50, max_n=1000):
    """
    计算案例(a)的解析解，动态决定需要的傅里叶项数以满足指定的容差，同时确保至少使用min_n项。
    对于大的n值，使用指数近似来避免数值溢出。

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
        n_approx_start: 开始使用近似的n值
    """
    phi = np.zeros_like(x)
    n_used = 0  # 实际使用的最大n值
    n_approx_start = None  # 开始使用近似的n值

    sinh_threshold = 700  # 根据经验设定的阈值，可以调整

    for n in range(1, max_n + 1, 2):  # 仅奇数项
        n_pi_Ly_over_Lx = n * np.pi * Ly / Lx

        if n_pi_Ly_over_Lx < sinh_threshold:
            # 使用精确的sinh计算
            numerator = np.sinh(n * np.pi * y / Lx)
            denominator = np.sinh(n_pi_Ly_over_Lx)
            ratio = numerator / denominator
        else:
            # 使用指数近似
            if n_approx_start is None:
                n_approx_start = n
                print(f"在 n = {n} 时开始使用指数近似")
            ratio = np.exp(n * np.pi * (y - Ly) / Lx)

        term = 4 * V0 / (n * np.pi) * np.sin(n * np.pi * x / Lx) * ratio
        phi += term
        n_used = n  # 更新为当前的n值

        # 检查当前项的最大贡献是否小于容差，并且已经达到最小n值
        if n >= min_n and np.max(np.abs(term)) < tol:
            print(
                f"解析解在 n = {n} 项时达到收敛，最大项贡献为 {np.max(np.abs(term)):.2e}"
            )
            break
    else:
        print(
            f"解析解未在最大 n 值 {max_n} 内收敛，最后一项贡献为 {np.max(np.abs(term)):.2e}"
        )

    return phi, n_used, n_approx_start


def analytic_solution_case_b(
    X, Y, Lx, Ly, rho_epsilon0, tol=1e-30, max_n=1000, max_m=1000
):
    """改进的案例(b)解析解计算

    Returns:
        phi: 解析解
        n_max: 使用的最大奇数项
        m_max: 使用的最大奇数项
    """
    phi = np.zeros_like(X, dtype=np.float64)

    # 预计算归一化坐标来减少重复计算
    x_normalized = X / Lx
    y_normalized = Y / Ly

    # 存储所有项以便按大小排序
    terms = []
    n_max = 1
    m_max = 1

    # 只使用奇数项
    for n in range(1, max_n + 1, 2):
        for m in range(1, max_m + 1, 2):
            # 计算系数
            denominator = (n * np.pi / Lx) ** 2 + (m * np.pi / Ly) ** 2
            coefficient = 16 * rho_epsilon0 / (np.pi**2 * n * m * denominator)

            # 使用周期归约来计算正弦值，避免大参数导致的数值误差
            arg_x = np.remainder(n * x_normalized, 2)
            arg_y = np.remainder(m * y_normalized, 2)
            sin_term = np.sin(arg_x * np.pi) * np.sin(arg_y * np.pi)

            term = coefficient * sin_term
            max_contribution = np.max(np.abs(term))

            terms.append((max_contribution, term))
            n_max = n
            m_max = m

            if max_contribution < tol and n >= 10 and m >= 10:
                terms.sort(reverse=True, key=lambda x: x[0])
                for _, t in terms:
                    phi += t
                logging.info(
                    f"解析解在 n = {n}, m = {m} 项时达到收敛，最大项贡献为 {max_contribution:.2e}"
                )
                return phi, n_max, m_max

    logging.warning(f"解析解未在最大项数(N={max_n}, M={max_m})内收敛")
    terms.sort(reverse=True, key=lambda x: x[0])
    for _, term in terms:
        phi += term

    return phi, n_max, m_max


def analytic_solution_superposition(X, Y, Lx, Ly, rho_epsilon0, boundary_values):
    """使用叠加原理计算任意边界条件和均匀源项的解析解

    参数:
        X, Y: 网格坐标
        Lx, Ly: 区域尺寸
        rho_epsilon0: 源项系数
        boundary_values: 边界值字典 {"left":v1, "right":v2, "bottom":v3, "top":v4}

    返回:
        phi: 总电势
        n_used_max: 边界齐次解使用的最大项数
        n_approx_start: 开始使用指数近似的n值
        n_terms: 源项特解使用的最大n
        m_terms: 源项特解使用的最大m
    """
    # 1. 计算源项特解
    if abs(rho_epsilon0) > 1e-10:
        phi_particular, n_terms, m_terms = analytic_solution_case_b(
            X, Y, Lx, Ly, rho_epsilon0
        )
    else:
        phi_particular = np.zeros_like(X)
        n_terms = m_terms = None

    # 2. 计算边界齐次解
    phi_homogeneous = np.zeros_like(X)
    n_used_max = 0
    n_approx_start = None

    # 为每个非零边界计算齐次解
    for boundary, value in boundary_values.items():
        if abs(value) > 1e-10:
            # 注意：对每个边界，我们需要：
            # 1. 正确变换坐标系使得该边界成为"顶边"
            # 2. 正确对应边界值的符号
            if boundary == "right":
                # 将右边变为"顶边"：旋转90度逆时针
                phi_temp, n_used, n_approx = analytic_solution_case_a(
                    Y, X, Ly, Lx, value
                )
            elif boundary == "left":
                # 将左边变为"顶边"：旋转90度顺时针
                phi_temp, n_used, n_approx = analytic_solution_case_a(
                    Y, Lx - X, Ly, Lx, value
                )
            elif boundary == "top":
                # 顶边不需要变换
                phi_temp, n_used, n_approx = analytic_solution_case_a(
                    X, Y, Lx, Ly, value
                )
            else:  # bottom
                # 底边需要翻转Y坐标
                phi_temp, n_used, n_approx = analytic_solution_case_a(
                    X, Ly - Y, Lx, Ly, value
                )

            phi_homogeneous += phi_temp
            n_used_max = max(n_used_max, n_used)
            if n_approx is not None:
                n_approx_start = (
                    n_approx
                    if n_approx_start is None
                    else min(n_approx_start, n_approx)
                )

    return (
        phi_homogeneous + phi_particular,
        n_used_max,
        n_approx_start,
        n_terms,
        m_terms,
    )


class PoissonSolver:
    def __init__(
        self,
        Lx: float,
        Ly: float,
        Nx: int,
        Ny: int,
        rho_epsilon0: np.ndarray,
        boundary_values: dict,
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
        self.rho_epsilon0 = rho_epsilon0
        self.boundary_values = boundary_values

        # 计算网格间距
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)

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
        """计算SOR方法的最优松弛因子,参考numerical recipes 2nd 19.5.19"""
        rho_jacobi = (
            np.cos(np.pi / self.Nx) + np.cos(np.pi / self.Ny) * (self.dx / self.dy) ** 2
        ) / (1 + (self.dx / self.dy) ** 2)
        omega = 2 / (1 + np.sqrt(1 - rho_jacobi**2))
        return omega

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

        # 计算系数
        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2 * (dx2 + dy2)

        for it in range(max_iter):
            phi_new[1:-1, 1:-1] = (
                (self.phi[2:, 1:-1] + self.phi[:-2, 1:-1]) * dy2
                + (self.phi[1:-1, 2:] + self.phi[1:-1, :-2]) * dx2
                + dx2 * dy2 * self.rho_epsilon0[1:-1, 1:-1]
            ) / denom

            max_change = np.max(np.abs(phi_new - self.phi))
            history.append(max_change)

            if save_history and it % history_interval == 0:
                self.phi_history.append((it, np.copy(self.phi)))

            self.phi, phi_new = phi_new, self.phi  # 交换变量

            if max_change < tolerance:
                print(f"Jacobi方法在第{it+1}次迭代收敛")
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

        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2 * (dx2 + dy2)

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    old_value = self.phi[i, j]
                    self.phi[i, j] = (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) * dy2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) * dx2
                        + dx2 * dy2 * self.rho_epsilon0[i, j]
                    ) / denom
                    max_change = max(max_change, abs(self.phi[i, j] - old_value))

            history.append(max_change)

            if save_history and it % history_interval == 0:
                self.phi_history.append((it, np.copy(self.phi)))

            if max_change < tolerance:
                print(f"Gauss-Seidel方法在第{it+1}次迭代收敛")
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

        dx2 = self.dx**2
        dy2 = self.dy**2
        denom = 2 * (dx2 + dy2)

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    old_value = self.phi[i, j]
                    phi_new = (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) * dy2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) * dx2
                        + dx2 * dy2 * self.rho_epsilon0[i, j]
                    ) / denom
                    self.phi[i, j] = (1 - omega) * old_value + omega * phi_new
                    max_change = max(max_change, abs(self.phi[i, j] - old_value))

            history.append(max_change)

            if save_history and it % history_interval == 0:
                self.phi_history.append((it, np.copy(self.phi)))

            if max_change < tolerance:
                print(f"SOR方法在第{it+1}次迭代收敛")
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
    contour = ax.contourf(X, Y, phi_matrices[0], levels=levels, cmap=cmap)

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
            cmap=cmap,
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


def plot_convergence_curves(results: dict):
    """在一个子图中绘制所有方法的收敛曲线"""
    methods = ["jacobi", "gauss_seidel", "sor"]
    _, ax = plt.subplots(figsize=(8, 6))

    for method in methods:
        history = results[method]["history"]
        iterations = range(1, len(history) + 1)
        ax.plot(iterations, history, label=method.upper())

    ax.set_xlabel("迭代次数")
    ax.set_ylabel("最大变化量")
    ax.set_title("各方法的收敛曲线")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

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
    n_approx_start: Optional[int] = None,
    n_terms: Optional[int] = None,
    m_terms: Optional[int] = None,
):
    """绘制最终结果对比图及差异分布图"""
    num_methods = len(results)
    fig, axs = plt.subplots(num_methods + 1, 2, figsize=(12, 4 * (num_methods + 1)))

    def setup_subplot(ax, phi, title, cmap=cmap, is_diff=False):
        mesh = ax.pcolormesh(X, Y, phi, cmap=cmap, shading="gouraud")
        plt.colorbar(mesh, ax=ax, label="差异 (V)" if is_diff else "电势 (V)")

        # 自定义鼠标悬停显示格式
        def format_coord(x, y):
            i = np.searchsorted(X[:, 0], x) - 1
            j = np.searchsorted(Y[0, :], y) - 1
            if 0 <= i < X.shape[0] and 0 <= j < Y.shape[1]:
                z = phi[i, j]
                if is_diff:
                    return f"x={x:.3f}, y={y:.3f}, Δφ={z:.3e}V"
                return f"x={x:.3f}, y={y:.3f}, φ={z:.3e}V"
            return f"x={x:.3f}, y={y:.3f}"

        ax.format_coord = format_coord

        # 添加边界标注和线条
        if not is_diff:
            # 添加带阴影背景的边界标注
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

        # 添加边界线
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=problem_params["Ly"], color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(x=problem_params["Lx"], color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title, pad=15)

    # 绘制解析解
    if phi_analytic is not None:
        setup_subplot(axs[0, 0], phi_analytic, "解析解")
    else:
        axs[0, 0].text(0.5, 0.5, "无解析解", ha="center", va="center", fontsize=16)
        axs[0, 0].set_title("解析解")

    ax_annotation = axs[0, 1]
    ax_annotation.axis("off")
    annotation_text = "解析解公式:\n"
    if case_name == "案例(a)":
        formula = r"$\phi(x,y) = \frac{4V_0}{\pi} \sum_{n=1,3,5,\dots}^{N} \frac{1}{n} \sin\left(\frac{n\pi x}{L_x}\right) \frac{\sinh\left(\frac{n\pi y}{L_x}\right)}{\sinh\left(\frac{n\pi L_y}{L_x}\right)}$"
        annotation_text += f"{formula}\n\n使用的傅里叶项:\n"
        annotation_text += f"最大 n = {n_used}\n"
        if n_approx_start is not None:
            annotation_text += f"从 n = {n_approx_start} 开始使用指数近似"
    elif case_name == "案例(b)":
        # 修改为只包含奇数项，并修正π的幂次为2
        formula = r"$\phi(x,y) = \sum_{n=1,3,5,\dots}^{N} \sum_{m=1,3,5,\dots}^{M} \frac{16 \rho}{\varepsilon_0 \pi^2 n m \left( \left( \frac{n\pi}{L_x} \right)^2 + \left( \frac{m\pi}{L_y} \right)^2 \right)} \sin\left( \frac{n\pi x}{L_x} \right) \sin\left( \frac{m\pi y}{L_y} \right)$"
        annotation_text += f"{formula}\n\n使用的傅里叶项:\n"
        annotation_text += f"最大奇数项: N = {n_terms}, M = {m_terms}"
    else:  # 自定义案例
        if abs(problem_params["rho_epsilon0"]) > 1e-10:
            annotation_text += "特解（源项）：\n"
            # 修改特解公式为只包含奇数项
            formula = r"$\phi_p(x,y) = \sum_{n=1,3,5,\dots}^{N} \sum_{m=1,3,5,\dots}^{M} \frac{16 \rho}{\varepsilon_0 \pi^2 n m \left( \left( \frac{n\pi}{L_x} \right)^2 + \left( \frac{m\pi}{L_y} \right)^2 \right)} \sin\left( \frac{n\pi x}{L_x} \right) \sin\left( \frac{m\pi y}{L_y} \right)$"
            annotation_text += f"{formula}\n"
            if n_terms and m_terms:
                annotation_text += (
                    f"\n特解使用的最大奇数项: N = {n_terms}, M = {m_terms}"
                )

        if any(abs(v) > 1e-10 for v in problem_params["boundary_values"].values()):
            if abs(problem_params["rho_epsilon0"]) > 1e-10:
                annotation_text += "\n\n"
            annotation_text += "齐次解（边界条件）：\n"
            formula = r"$\phi_h(x,y) = \sum_{i=1}^{4} \frac{4V_i}{\pi} \sum_{n=1,3,5,\dots}^{N} \frac{1}{n} \sin\left(\frac{n\pi x_i}{L_x}\right) \frac{\sinh\left(\frac{n\pi y_i}{L_x}\right)}{\sinh\left(\frac{n\pi L_y}{L_x}\right)}$"
            annotation_text += f"{formula}\n"
            if n_used:
                annotation_text += f"\n齐次解使用的最大傅里叶项: n = {n_used}"
            if n_approx_start:
                annotation_text += f"\n从 n = {n_approx_start} 开始使用指数近似"

    ax_annotation.text(
        0.5,
        0.5,
        annotation_text,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

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


def solve_and_visualize(problem_params: dict, case_name: str):
    """求解并可视化结果"""
    # 设置网格
    Nx = problem_params["Nx"]
    Ny = problem_params["Ny"]
    x = np.linspace(0, problem_params["Lx"], Nx)
    y = np.linspace(0, problem_params["Ly"], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 根据案例选择计算解析解
    if case_name == "案例(a)":
        # 计算案例(a)的解析解
        V0 = problem_params["boundary_values"]["top"]
        phi_analytic, n_used, n_approx_start = analytic_solution_case_a(
            X,
            Y,
            problem_params["Lx"],
            problem_params["Ly"],
            V0,
            tol=1e-16,
            min_n=50,
            max_n=1000,
        )
        print(f"解析解使用的傅里叶级数中最大 n: {n_used}")
        if n_approx_start:
            print(f"从 n = {n_approx_start} 开始使用指数近似")
        n_terms = n_used
        m_terms = None  # 案例(a)只有一个方向的傅里叶展开
    elif case_name == "案例(b)":
        # 计算案例(b)的解析解
        rho_epsilon0 = problem_params["rho_epsilon0"]
        phi_analytic, n_terms, m_terms = analytic_solution_case_b(
            X,
            Y,
            problem_params["Lx"],
            problem_params["Ly"],
            rho_epsilon0,
        )
        print(f"解析解使用的傅里叶级数项数: N = {n_terms}, M = {m_terms}")
        n_used = n_terms
        n_approx_start = None
    else:  # 自定义案例
        # 使用叠加原理计算解析解
        phi_analytic, n_used, n_approx_start, n_terms, m_terms = (
            analytic_solution_superposition(
                X,
                Y,
                problem_params["Lx"],
                problem_params["Ly"],
                problem_params["rho_epsilon0"],
                problem_params["boundary_values"],
            )
        )
        print(f"解析解计算完成")
        if n_terms and m_terms:
            print(f"特解使用的最大奇数项: N = {n_terms}, M = {m_terms}")
        if n_used:
            print(f"齐次解使用的最大傅里叶项: n = {n_used}")
        if n_approx_start:
            print(f"从 n = {n_approx_start} 开始使用指数近似")

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
            max_iter=problem_params["max_iter"],
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

    # 绘制收敛曲线
    plot_convergence_curves(results)

    # 绘制最终结果对比图
    if case_name == "案例(a)":
        plot_final_results(
            X,
            Y,
            results,
            phi_analytic,
            case_name,
            problem_params,
            n_used,
            n_approx_start,
        )
    elif case_name == "案例(b)":
        plot_final_results(
            X,
            Y,
            results,
            phi_analytic,
            case_name,
            problem_params,
            n_terms=n_terms,
            m_terms=m_terms,
        )
    else:  # 自定义案例
        plot_final_results(
            X,
            Y,
            results,
            phi_analytic,
            case_name,
            problem_params,
            n_used=n_used,
            n_approx_start=n_approx_start,
            n_terms=n_terms,
            m_terms=m_terms,
        )


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
        case_name = "案例(a)"
    elif case == "b":
        problem_params = {
            "Lx": 1.0,
            "Ly": 1.0,
            "rho_epsilon0": 1.0,
            "boundary_values": {"left": 0, "right": 0, "bottom": 0, "top": 0},
        }
        case_name = "案例(b)"
    else:  # case == 'c'
        print("\n=== 请输入自定义参数 ===")
        print("（注：所有长度单位为米(m)，电势单位为伏特(V)）")

        # 获取区域尺寸
        while True:
            try:
                Lx = float(input("请输入 x 方向长度 Lx: "))
                Ly = float(input("请输入 y 方向长度 Ly: "))
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
                    value = float(input(f"{boundary} 边界电势: "))
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
        case_name = "自定义案例"

    # 获取网格和迭代参数
    print("\n请输入网格点数 (Nx, Ny) 和最大迭代次数 (max_iter)，按回车使用默认值：")

    # 计算默认的 Nx 和 Ny
    default_Nx = int(50 * problem_params["Lx"])
    default_Ny = int(50 * problem_params["Ly"])
    default_max_iter = 10000

    # 获取 Nx
    while True:
        nx_input = input(f"请输入 x 方向的网格点数 Nx (默认 {default_Nx}): ").strip()
        if nx_input == "":
            Nx = default_Nx  # 默认值
            break
        else:
            try:
                Nx = int(nx_input)
                if Nx > 0:
                    break
                else:
                    print("Nx 必须为正整数！")
            except ValueError:
                print("请输入有效的整数！")

    # 获取 Ny
    while True:
        ny_input = input(f"请输入 y 方向的网格点数 Ny (默认 {default_Ny}): ").strip()
        if ny_input == "":
            Ny = default_Ny  # 默认值
            break
        else:
            try:
                Ny = int(ny_input)
                if Ny > 0:
                    break
                else:
                    print("Ny 必须为正整数！")
            except ValueError:
                print("请输入有效的整数！")

    # 获取 max_iter
    while True:
        max_iter_input = input(
            f"请输入最大迭代次数 max_iter (默认 {default_max_iter}): "
        ).strip()
        if max_iter_input == "":
            max_iter = default_max_iter  # 默认值
            break
        else:
            try:
                max_iter = int(max_iter_input)
                if max_iter > 0:
                    break
                else:
                    print("max_iter 必须为正整数！")
            except ValueError:
                print("请输入有效的整数！")

    # 将这些参数添加到 problem_params 中
    problem_params["Nx"] = Nx
    problem_params["Ny"] = Ny
    problem_params["max_iter"] = max_iter

    return problem_params, case_name


def main():
    """主程序"""
    # 配置matplotlib中文字体
    configure_matplotlib_fonts()

    # 获取用户输入并求解
    problem_params, case_name = get_user_case()
    solve_and_visualize(problem_params, case_name)


if __name__ == "__main__":
    main()
