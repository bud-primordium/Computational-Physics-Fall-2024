import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import logging
from typing import Tuple, List, Optional
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def configure_matplotlib_fonts():
    """配置matplotlib的字体设置，支持中文显示"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 检测系统并设置中文字体
    system = platform.system()
    font_families = {
        "Darwin": ["Arial Unicode MS"],  # macOS
        "Windows": ["Microsoft YaHei"],  # Windows
        "Linux": ["WenQuanYi Micro Hei"],  # Linux
    }

    if system in font_families:
        plt.rcParams["font.family"] = font_families[system]

    # 尝试设置备选字体
    try:
        plt.rcParams["font.sans-serif"] = [
            "Arial Unicode MS",
            "SimSun",
            "STSong",
            "SimHei",
        ] + plt.rcParams["font.sans-serif"]
    except:
        logger.warning("未能设置理想的中文字体，尝试使用系统默认字体")

    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False


def analytic_solution_case_a(x, y, Lx, Ly, V0, n_terms=50):
    """
    解析解用于案例(a):
    -∇²φ = 0
    φ = 0 在 x=0, x=Lx, y=0
    φ = V0 在 y=Ly
    """
    phi = np.zeros_like(x)
    for n in range(1, n_terms, 2):  # 只取奇数项
        phi += (
            4
            * V0
            / (n * np.pi)
            * np.sin(n * np.pi * x / Lx)
            * np.sinh(n * np.pi * y / Lx)
            / np.sinh(n * np.pi * Ly / Lx)
        )
    return phi


def add_boundary_annotations(ax, Lx: float, Ly: float, boundary_values: dict):
    """在图上添加边界条件标注"""
    # 添加边界电势值标注
    ax.text(
        -0.1,
        Ly / 2,
        f"φ = {boundary_values['left']}V",
        rotation=90,
        verticalalignment="center",
    )
    ax.text(
        Lx + 0.1,
        Ly / 2,
        f"φ = {boundary_values['right']}V",
        rotation=-90,
        verticalalignment="center",
    )
    ax.text(
        Lx / 2, -0.1, f"φ = {boundary_values['bottom']}V", horizontalalignment="center"
    )
    ax.text(
        Lx / 2, Ly + 0.1, f"φ = {boundary_values['top']}V", horizontalalignment="center"
    )

    # 用虚线标出边界
    ax.plot([0, Lx], [0, 0], "k--", alpha=0.5)  # 底边
    ax.plot([0, Lx], [Ly, Ly], "k--", alpha=0.5)  # 顶边
    ax.plot([0, 0], [0, Ly], "k--", alpha=0.5)  # 左边
    ax.plot([Lx, Lx], [0, Ly], "k--", alpha=0.5)  # 右边


def create_convergence_title(rho_epsilon0: float) -> str:
    """创建包含源项信息的标题"""
    if abs(rho_epsilon0) < 1e-10:
        return "无源项 (ρ = 0)"
    else:
        return f"均匀源项 (ρ/ε₀ = {rho_epsilon0} V/m²)"


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
        初始化泊松方程求解器

        参数:
            Lx, Ly: 计算区域的长度和宽度 (m)
            Nx, Ny: x和y方向的网格点数
            rho: 电荷密度矩阵 (C/m³)
            boundary_values: 边界条件字典，包含 'left', 'right', 'top', 'bottom' 的值
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

        # 用于存储迭代过程中的电势分布
        self.phi_history = []

    def set_boundary_conditions(self):
        """设置边界条件"""
        self.phi[:, 0] = self.boundary_values["bottom"]  # 底边界
        self.phi[:, -1] = self.boundary_values["top"]  # 顶边界
        self.phi[0, :] = self.boundary_values["left"]  # 左边界
        self.phi[-1, :] = self.boundary_values["right"]  # 右边界

    def get_optimal_omega(self) -> float:
        """
        计算SOR方法的理论最优松弛因子
        ω_opt = 2/(1 + sin(π/N))，其中N是最大网格数
        """
        N = max(self.Nx, self.Ny)
        return 2 / (1 + np.sin(np.pi / N))

    def theoretical_iteration_count(self, method: str, error_reduction: float) -> int:
        """
        计算理论迭代次数

        参数:
            method: 求解方法 ('jacobi', 'gauss_seidel', 或 'sor')
            error_reduction: 误差衰减系数（如想降低到1e-6，则为6）

        返回:
            理论迭代次数
        """
        L = max(self.Nx, self.Ny)
        if method == "jacobi":
            return int(0.5 * error_reduction * L**2)
        elif method == "gauss_seidel":
            return int(0.25 * error_reduction * L**2)
        elif method == "sor":
            return int((1 / 3) * error_reduction * L)
        else:
            raise ValueError(f"未知的方法: {method}")

    def solve(
        self,
        method: str = "sor",
        tolerance: float = 1e-6,
        max_iter: int = 10000,
        save_history: bool = False,
        history_interval: int = 10,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """
        求解泊松方程

        参数:
            method: 求解方法
            tolerance: 收敛容差
            max_iter: 最大迭代次数
            save_history: 是否保存迭代历史
            history_interval: 保存历史的间隔步数

        返回:
            (phi, history, iterations, elapsed_time)
        """
        if method == "jacobi":
            return self.jacobi(tolerance, max_iter, save_history, history_interval)
        elif method == "gauss_seidel":
            return self.gauss_seidel(
                tolerance, max_iter, save_history, history_interval
            )
        elif method == "sor":
            omega = self.get_optimal_omega()
            print(f"使用理论最优松弛因子: ω = {omega:.3f}")
            return self.sor(omega, tolerance, max_iter, save_history, history_interval)
        else:
            raise ValueError(f"未知的求解方法: {method}")

    def jacobi(
        self,
        tolerance: float = 1e-6,
        max_iter: int = 10000,
        save_history: bool = False,
        history_interval: int = 10,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """Jacobi迭代法"""
        phi_new = np.copy(self.phi)
        history = []
        start_time = time.time()

        # 预计算常数项
        constant = 1 / (2 * (1 / self.dx**2 + 1 / self.dy**2))
        rho_term = self.rho / self.epsilon0

        for it in range(max_iter):
            # 更新内部点
            phi_new[1:-1, 1:-1] = constant * (
                (self.phi[2:, 1:-1] + self.phi[:-2, 1:-1]) / self.dx**2
                + (self.phi[1:-1, 2:] + self.phi[1:-1, :-2]) / self.dy**2
                - rho_term[1:-1, 1:-1]
            )

            # 计算最大变化量
            max_change = np.max(np.abs(phi_new - self.phi))
            history.append(max_change)

            # 保存迭代历史
            if save_history and it % history_interval == 0:
                self.phi_history.append(np.copy(self.phi))

            # 更新解
            self.phi, phi_new = phi_new.copy(), self.phi

            # 检查收敛
            if max_change < tolerance:
                print(f"Jacobi方法在第{it}次迭代收敛")
                break
        else:
            print("Jacobi方法在最大迭代次数内未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def gauss_seidel(
        self,
        tolerance: float = 1e-6,
        max_iter: int = 10000,
        save_history: bool = False,
        history_interval: int = 10,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """Gauss-Seidel迭代法"""
        history = []
        start_time = time.time()

        # 预计算常数项
        constant = 1 / (2 * (1 / self.dx**2 + 1 / self.dy**2))
        rho_term = self.rho / self.epsilon0

        for it in range(max_iter):
            max_change = 0
            # 更新内部点
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_old = self.phi[i, j]
                    self.phi[i, j] = constant * (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) / self.dx**2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) / self.dy**2
                        - rho_term[i, j]
                    )
                    max_change = max(max_change, abs(self.phi[i, j] - phi_old))

            history.append(max_change)

            # 保存迭代历史
            if save_history and it % history_interval == 0:
                self.phi_history.append(np.copy(self.phi))

            # 检查收敛
            if max_change < tolerance:
                print(f"Gauss-Seidel方法在第{it}次迭代收敛")
                break
        else:
            print("Gauss-Seidel方法在最大迭代次数内未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def sor(
        self,
        omega: float,
        tolerance: float = 1e-6,
        max_iter: int = 10000,
        save_history: bool = False,
        history_interval: int = 10,
    ) -> Tuple[np.ndarray, List[float], int, float]:
        """SOR迭代法"""
        history = []
        start_time = time.time()

        # 预计算常数项
        constant = 1 / (2 * (1 / self.dx**2 + 1 / self.dy**2))
        rho_term = self.rho / self.epsilon0

        for it in range(max_iter):
            max_change = 0
            # 更新内部点
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_old = self.phi[i, j]
                    phi_new = constant * (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) / self.dx**2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) / self.dy**2
                        - rho_term[i, j]
                    )
                    self.phi[i, j] = (1 - omega) * phi_old + omega * phi_new
                    max_change = max(max_change, abs(self.phi[i, j] - phi_old))

            history.append(max_change)

            # 保存迭代历史
            if save_history and it % history_interval == 0:
                self.phi_history.append(np.copy(self.phi))

            # 检查收敛
            if max_change < tolerance:
                print(f"SOR方法在第{it}次迭代收敛")
                break
        else:
            print("SOR方法在最大迭代次数内未收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time


def get_user_case() -> Tuple[dict, str]:
    """
    获取用户选择的求解案例

    返回:
        problem_params: 问题参数字典
        case_name: 案例名称
    """
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


def animate_solution(X, Y, phi_history, method_name, problem_params):
    """创建迭代过程动画"""
    fig, ax = plt.subplots()
    levels = np.linspace(
        np.min(phi_history[-1]), np.max(phi_history[-1]), 50
    )  # 固定颜色范围
    contour = ax.contourf(X, Y, phi_history[0], levels=levels, cmap="viridis")
    plt.colorbar(contour, ax=ax)
    add_boundary_annotations(
        ax,
        problem_params["Lx"],
        problem_params["Ly"],
        problem_params["boundary_values"],
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{method_name} 方法迭代过程")

    def update(i):
        ax.clear()
        contour = ax.contourf(X, Y, phi_history[i], levels=levels, cmap="viridis")
        add_boundary_annotations(
            ax,
            problem_params["Lx"],
            problem_params["Ly"],
            problem_params["boundary_values"],
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{method_name} 方法迭代过程\n第 {i} 次迭代")

    anim = FuncAnimation(fig, update, frames=len(phi_history), interval=200)
    plt.show()


def solve_and_visualize(problem_params: dict, case_name: str):
    """求解并可视化结果"""
    Nx, Ny = 50, 50  # 网格点数
    x = np.linspace(0, problem_params["Lx"], Nx)
    y = np.linspace(0, problem_params["Ly"], Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # 计算解析解（仅针对案例(a)）
    if (
        problem_params["rho_epsilon0"] == 0
        and problem_params["boundary_values"]["top"] != 0
    ):
        V0 = problem_params["boundary_values"]["top"]
        phi_analytic = analytic_solution_case_a(
            X, Y, problem_params["Lx"], problem_params["Ly"], V0
        )
    else:
        phi_analytic = np.zeros_like(X)
        print("解析解未实现，仅显示数值解")

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
            method=method, save_history=True, history_interval=10
        )
        if np.max(np.abs(phi_analytic)) > 0:
            error = np.max(np.abs(phi - phi_analytic)) / np.max(np.abs(phi_analytic))
        else:
            error = None

        # 计算理论迭代次数（对于误差降至1e-6）
        theoretical_iter = solver.theoretical_iteration_count(method, 6)

        results[method] = {
            "phi": phi,
            "history": history,
            "iterations": iterations,
            "elapsed": elapsed,
            "error": error,
            "theoretical_iter": theoretical_iter,
            "phi_history": solver.phi_history,
        }

        print(f"迭代次数: {iterations} (理论预期: {theoretical_iter})")
        print(f"求解时间: {elapsed:.4f}秒")
        if error is not None:
            print(f"相对误差: {error:.2e}")

        # 动画展示
        animate_solution(X, Y, solver.phi_history, method.upper(), problem_params)

    # 创建2x2子图布局
    _, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 解析解
    ax1 = axes[0, 0]
    if np.max(np.abs(phi_analytic)) > 0:
        im1 = ax1.contourf(X, Y, phi_analytic, levels=50, cmap="viridis")
        plt.colorbar(im1, ax=ax1, label="电势 (V)")
        add_boundary_annotations(
            ax1,
            problem_params["Lx"],
            problem_params["Ly"],
            problem_params["boundary_values"],
        )
        ax1.set_title("解析解")
    else:
        ax1.text(0.5, 0.5, "解析解未实现", ha="center", va="center", fontsize=16)
        ax1.set_title("解析解")

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    # 2-4. 数值解
    for i, method in enumerate(methods):
        ax = axes.flatten()[i + 1]
        im = ax.contourf(X, Y, results[method]["phi"], levels=50, cmap="viridis")
        plt.colorbar(im, ax=ax, label="电势 (V)")
        add_boundary_annotations(
            ax,
            problem_params["Lx"],
            problem_params["Ly"],
            problem_params["boundary_values"],
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        title = f"{method.upper()} 方法"
        if results[method]["error"] is not None:
            title += f"\n相对误差: {results[method]['error']:.2e}"
        ax.set_title(title)

    # 添加总标题
    plt.suptitle(
        f'{case_name}\n{create_convergence_title(problem_params["rho_epsilon0"])}',
        fontsize=14,
        y=0.95,
    )
    plt.tight_layout()
    plt.show()


def main():
    """主程序"""
    # 配置matplotlib中文字体
    configure_matplotlib_fonts()

    # 获取用户输入并求解
    problem_params, case_name = get_user_case()
    solve_and_visualize(problem_params, case_name)


if __name__ == "__main__":
    main()
