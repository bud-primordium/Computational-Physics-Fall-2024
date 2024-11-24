import numpy as np
import matplotlib.pyplot as plt
import time
import platform
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 检测系统并设置中文字体
system = platform.system()

# 根据操作系统设置中文字体
if system == "Darwin":  # macOS
    plt.rcParams["font.family"] = ["Arial Unicode MS"]
elif system == "Windows":
    plt.rcParams["font.family"] = ["Microsoft YaHei"]
elif system == "Linux":
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]

# 如果上述字体都不可用，尝试使用系统默认字体
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


class PoissonSolver:
    def __init__(self, Lx, Ly, Nx, Ny, rho, boundary_conditions, epsilon0=8.854e-12):
        """
        初始化泊松方程求解器。

        参数:
            Lx, Ly: 区域长度（米）
            Nx, Ny: 网格点数
            rho: 电荷密度矩阵
            boundary_conditions: 边界条件字典，包含 'left', 'right', 'bottom', 'top' 的电势值
            epsilon0: 真空介电常数
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.rho = rho
        self.boundary_conditions = boundary_conditions
        self.epsilon0 = epsilon0

        self.dx = Lx / (Nx - 1)
        self.dy = Ly / (Ny - 1)
        self.phi = np.zeros((Nx, Ny))

        # 设置边界条件
        self.set_boundary_conditions()

    def set_boundary_conditions(self):
        """
        根据边界条件字典设置电势矩阵的边界。
        """
        self.phi[:, 0] = self.boundary_conditions.get("bottom", 0)
        self.phi[:, -1] = self.boundary_conditions.get("top", 0)
        self.phi[0, :] = self.boundary_conditions.get("left", 0)
        self.phi[-1, :] = self.boundary_conditions.get("right", 0)

    def jacobi(self, tolerance=1e-6, max_iter=10000):
        """
        使用 Jacobi 方法求解泊松方程。

        本质：
            Jacobi 方法使用上一迭代步的所有电势值来更新下一迭代步的电势值。
            适合并行计算，但收敛速度较慢。

            更新公式：
            φ_i,j^(new) = [ (φ_(i+1,j) + φ_(i-1,j)) / Δx² + (φ_(i,j+1) + φ_(i,j-1)) / Δy² + ρ_i,j / ε₀ ] / [ 2 * (1/Δx² + 1/Δy²) ]

        返回:
            phi: 最终电势分布
            history: 每次迭代的最大变化量
            iterations: 实际迭代次数
            elapsed_time: 求解时间
        """
        phi_new = self.phi.copy()
        history = []
        start_time = time.time()

        for it in range(max_iter):
            # 更新内部点
            phi_new[1:-1, 1:-1] = (
                (self.phi[2:, 1:-1] + self.phi[:-2, 1:-1]) / self.dx**2
                + (self.phi[1:-1, 2:] + self.phi[1:-1, :-2]) / self.dy**2
                + self.rho[1:-1, 1:-1] / self.epsilon0
            ) / (2 * (1 / self.dx**2 + 1 / self.dy**2))

            # 计算最大变化量
            max_change = np.max(np.abs(phi_new - self.phi))
            history.append(max_change)

            # 更新 phi
            self.phi[:, :] = phi_new[:, :]

            # 检查收敛
            if max_change < tolerance:
                print(f"Jacobi 方法在迭代次数 {it} 收敛")
                break
        else:
            print("Jacobi 方法未在最大迭代次数内收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def gauss_seidel(self, tolerance=1e-6, max_iter=10000):
        """
        使用 Gauss-Seidel 方法求解泊松方程。

        本质：
            Gauss-Seidel 方法在同一迭代步中立即使用已更新的电势值，加快收敛速度。
            不适合并行计算。

            更新公式：
            φ_i,j = [ (φ_(i+1,j) + φ_(i-1,j)) / Δx² + (φ_(i,j+1) + φ_(i,j-1)) / Δy² + ρ_i,j / ε₀ ] / [ 2 * (1/Δx² + 1/Δy²) ]

        返回:
            phi: 最终电势分布
            history: 每次迭代的最大变化量
            iterations: 实际迭代次数
            elapsed_time: 求解时间
        """
        history = []
        start_time = time.time()

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_old = self.phi[i, j]
                    self.phi[i, j] = (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) / self.dx**2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) / self.dy**2
                        + self.rho[i, j] / self.epsilon0
                    ) / (2 * (1 / self.dx**2 + 1 / self.dy**2))
                    change = abs(self.phi[i, j] - phi_old)
                    if change > max_change:
                        max_change = change
            history.append(max_change)
            if max_change < tolerance:
                print(f"Gauss-Seidel 方法在迭代次数 {it} 收敛")
                break
        else:
            print("Gauss-Seidel 方法未在最大迭代次数内收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time

    def sor(self, omega=1.5, tolerance=1e-6, max_iter=10000):
        """
        使用 SOR 方法求解泊松方程。

        本质：
            SOR 方法在 Gauss-Seidel 方法的基础上引入松弛因子 ω（1 < ω < 2），
            通过超松弛来加快收敛速度。

            更新公式：
            φ_i,j^(new) = (1 - ω) * φ_i,j^(old) + ω * [ (φ_(i+1,j) + φ_(i-1,j)) / Δx² + (φ_(i,j+1) + φ_(i,j-1)) / Δy² + ρ_i,j / ε₀ ] / [ 2 * (1/Δx² + 1/Δy²) ]

        参数:
            omega: 松弛因子（1 < omega < 2）

        返回:
            phi: 最终电势分布
            history: 每次迭代的最大变化量
            iterations: 实际迭代次数
            elapsed_time: 求解时间
        """
        history = []
        start_time = time.time()

        for it in range(max_iter):
            max_change = 0
            for i in range(1, self.Nx - 1):
                for j in range(1, self.Ny - 1):
                    phi_old = self.phi[i, j]
                    phi_new = (
                        (self.phi[i + 1, j] + self.phi[i - 1, j]) / self.dx**2
                        + (self.phi[i, j + 1] + self.phi[i, j - 1]) / self.dy**2
                        + self.rho[i, j] / self.epsilon0
                    ) / (2 * (1 / self.dx**2 + 1 / self.dy**2))
                    self.phi[i, j] = (1 - omega) * phi_old + omega * phi_new
                    change = abs(self.phi[i, j] - phi_old)
                    if change > max_change:
                        max_change = change
            history.append(max_change)
            if max_change < tolerance:
                print(f"SOR 方法在迭代次数 {it} 收敛")
                break
        else:
            print("SOR 方法未在最大迭代次数内收敛")

        elapsed_time = time.time() - start_time
        return self.phi, history, it + 1, elapsed_time


def plot_potential(phi, Lx, Ly, title="电势分布"):
    """
    绘制电势分布的等高线图。

    参数:
        phi: 电势矩阵
        Lx, Ly: 区域长度
        title: 图表标题
    """
    Nx, Ny = phi.shape
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, phi, levels=50, cmap="viridis")
    plt.colorbar(cp, label="电势 (V)")
    plt.xlabel("x (米)")
    plt.ylabel("y (米)")
    plt.title(title)
    plt.show()


def plot_convergence(history, title="收敛曲线"):
    """
    绘制收敛曲线。

    参数:
        history: 每次迭代的最大变化量
        title: 图表标题
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history, label="最大变化量")
    plt.yscale("log")
    plt.xlabel("迭代次数")
    plt.ylabel("最大变化量")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def example_a():
    """
    示例(a)：
    无源项，特定边界条件：
    - ρ(x, y) = 0
    - φ(0, y) = φ(Lx, y) = φ(x, 0) = 0
    - φ(x, Ly) = 1 V
    - Lx = 1 m, Ly = 1.5 m
    """
    print("运行示例(a)：无源项，特定边界条件")
    Lx, Ly = 1.0, 1.5
    Nx, Ny = 50, 75
    rho = np.zeros((Nx, Ny))  # 情况(a): rho = 0
    boundary_conditions = {
        "left": 0,
        "right": 0,
        "bottom": 0,
        "top": 1,  # 情况(a): y = Ly 边界电势为1V
    }

    solver = PoissonSolver(Lx, Ly, Nx, Ny, rho, boundary_conditions)

    # 选择方法
    methods = ["jacobi", "gauss_seidel", "sor"]
    results = {}

    for method in methods:
        if method == "jacobi":
            phi, history, iterations, elapsed = solver.jacobi()
        elif method == "gauss_seidel":
            phi, history, iterations, elapsed = solver.gauss_seidel()
        elif method == "sor":
            phi, history, iterations, elapsed = solver.sor(omega=1.5)
        results[method] = (phi, history, iterations, elapsed)
        print(f"{method.capitalize()} 方法: {iterations} 次迭代, 用时 {elapsed:.4f} 秒")

    # 可视化电势分布和收敛曲线
    for method in methods:
        phi, history, iterations, elapsed = results[method]
        plot_potential(
            phi, Lx, Ly, title=f"示例(a) 电势分布 - {method.capitalize()} 方法"
        )
        plot_convergence(
            history, title=f"示例(a) 收敛曲线 - {method.capitalize()} 方法"
        )


def example_b():
    """
    示例(b)：
    有均匀源项，特定边界条件：
    - ρ(x, y)/ε0 = 1 V/m²
    - φ(0, y) = φ(Lx, y) = φ(x, 0) = φ(x, Ly) = 0
    - Lx = Ly = 1 m
    """
    print("运行示例(b)：有均匀源项，特定边界条件")
    Lx, Ly = 1.0, 1.0
    Nx, Ny = 50, 50
    rho = np.ones((Nx, Ny)) * 1.0  # ρ/epsilon0 = 1 V/m²
    boundary_conditions = {
        "left": 0,
        "right": 0,
        "bottom": 0,
        "top": 0,  # 情况(b): 所有边界电势为0V
    }

    solver = PoissonSolver(Lx, Ly, Nx, Ny, rho, boundary_conditions)

    # 选择方法
    methods = ["jacobi", "gauss_seidel", "sor"]
    results = {}

    for method in methods:
        if method == "jacobi":
            phi, history, iterations, elapsed = solver.jacobi()
        elif method == "gauss_seidel":
            phi, history, iterations, elapsed = solver.gauss_seidel()
        elif method == "sor":
            phi, history, iterations, elapsed = solver.sor(omega=1.5)
        results[method] = (phi, history, iterations, elapsed)
        print(f"{method.capitalize()} 方法: {iterations} 次迭代, 用时 {elapsed:.4f} 秒")

    # 可视化电势分布和收敛曲线
    for method in methods:
        phi, history, iterations, elapsed = results[method]
        plot_potential(
            phi, Lx, Ly, title=f"示例(b) 电势分布 - {method.capitalize()} 方法"
        )
        plot_convergence(
            history, title=f"示例(b) 收敛曲线 - {method.capitalize()} 方法"
        )


def main():
    """
    主函数，运行示例(a)和示例(b)。
    """
    example_a()
    example_b()


if __name__ == "__main__":
    main()
