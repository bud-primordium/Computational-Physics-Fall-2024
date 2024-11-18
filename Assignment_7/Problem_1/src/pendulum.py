import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipj


class Pendulum:
    def __init__(self, length=1.0, mass=1.0, gravity=9.81, theta0=0.5, omega0=0.0):
        """
        初始化简谐摆的参数。

        参数:
        - length (float): 摆长（m）。
        - mass (float): 小球质量（kg）。
        - gravity (float): 重力加速度（m/s²）。
        - theta0 (float): 初始角度（rad）。
        - omega0 (float): 初始角速度（rad/s）。
        """
        self.L = length
        self.m = mass
        self.g = gravity
        self.theta0 = theta0
        self.omega0 = omega0

    def derivatives(self, theta, omega):
        """
        计算系统的导数 dtheta/dt 和 domega/dt。

        参数:
        - theta (float): 当前角度（rad）。
        - omega (float): 当前角速度（rad/s）。

        返回:
        - dtheta_dt (float): 角度的导数（rad/s）。
        - domega_dt (float): 角速度的导数（rad/s²）。
        """
        dtheta_dt = omega
        domega_dt = -(self.g / self.L) * np.sin(theta)
        return dtheta_dt, domega_dt

    def compute_energy(self, theta, omega):
        """
        计算系统的总能量，包括动能和势能。

        参数:
        - theta (np.ndarray): 角度数组（rad）。
        - omega (np.ndarray): 角速度数组（rad/s）。

        返回:
        - energy (np.ndarray): 总能量数组（J）。
        """
        kinetic = 0.5 * self.m * (self.L * omega) ** 2
        potential = self.m * self.g * self.L * (1 - np.cos(theta))
        return kinetic + potential

    def euler(self, h, N):
        """
        使用欧拉法求解简谐摆的运动。

        参数:
        - h (float): 时间步长（s）。
        - N (int): 总步数。

        返回:
        - theta (np.ndarray): 角度数组（rad）。
        - omega (np.ndarray): 角速度数组（rad/s）。
        """
        theta = np.zeros(N + 1)
        omega = np.zeros(N + 1)
        theta[0] = self.theta0
        omega[0] = self.omega0

        for i in range(N):
            dtheta_dt, domega_dt = self.derivatives(theta[i], omega[i])
            theta[i + 1] = theta[i] + h * dtheta_dt
            omega[i + 1] = omega[i] + h * domega_dt

        return theta, omega

    def midpoint(self, h, N):
        """
        使用中点法求解简谐摆的运动。

        参数:
        - h (float): 时间步长（s）。
        - N (int): 总步数。

        返回:
        - theta (np.ndarray): 角度数组（rad）。
        - omega (np.ndarray): 角速度数组（rad/s）。
        """
        theta = np.zeros(N + 1)
        omega = np.zeros(N + 1)
        theta[0] = self.theta0
        omega[0] = self.omega0

        for i in range(N):
            # 计算中点的斜率
            dtheta_dt, domega_dt = self.derivatives(theta[i], omega[i])
            theta_mid = theta[i] + 0.5 * h * dtheta_dt
            omega_mid = omega[i] + 0.5 * h * domega_dt

            # 使用中点斜率更新
            dtheta_dt_mid, domega_dt_mid = self.derivatives(theta_mid, omega_mid)
            theta[i + 1] = theta[i] + h * dtheta_dt_mid
            omega[i + 1] = omega[i] + h * domega_dt_mid

        return theta, omega

    def rk4(self, h, N):
        """
        使用四阶龙格-库塔法 (RK4) 求解简谐摆的运动。

        参数:
        - h (float): 时间步长（s）。
        - N (int): 总步数。

        返回:
        - theta (np.ndarray): 角度数组（rad）。
        - omega (np.ndarray): 角速度数组（rad/s）。
        """
        theta = np.zeros(N + 1)
        omega = np.zeros(N + 1)
        theta[0] = self.theta0
        omega[0] = self.omega0

        for i in range(N):
            # 计算k1
            k1_theta, k1_omega = self.derivatives(theta[i], omega[i])

            # 计算k2
            k2_theta, k2_omega = self.derivatives(
                theta[i] + 0.5 * h * k1_theta, omega[i] + 0.5 * h * k1_omega
            )

            # 计算k3
            k3_theta, k3_omega = self.derivatives(
                theta[i] + 0.5 * h * k2_theta, omega[i] + 0.5 * h * k2_omega
            )

            # 计算k4
            k4_theta, k4_omega = self.derivatives(
                theta[i] + h * k3_theta, omega[i] + h * k3_omega
            )

            # 更新角度和角速度
            theta[i + 1] = theta[i] + (h / 6.0) * (
                k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta
            )
            omega[i + 1] = omega[i] + (h / 6.0) * (
                k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega
            )

        return theta, omega

    def euler_trapezoidal(self, h, N):
        """
        使用欧拉-梯形法求解简谐摆的运动。

        参数:
        - h (float): 时间步长（s）。
        - N (int): 总步数。

        返回:
        - theta (np.ndarray): 角度数组（rad）。
        - omega (np.ndarray): 角速度数组（rad/s）。
        """
        theta = np.zeros(N + 1)
        omega = np.zeros(N + 1)
        theta[0] = self.theta0
        omega[0] = self.omega0

        for i in range(N):
            # 预测步骤（欧拉法）
            dtheta_dt, domega_dt = self.derivatives(theta[i], omega[i])
            theta_pred = theta[i] + h * dtheta_dt
            omega_pred = omega[i] + h * domega_dt

            # 校正步骤（梯形法）
            dtheta_dt_pred, domega_dt_pred = self.derivatives(theta_pred, omega_pred)
            theta[i + 1] = theta[i] + (h / 2.0) * (dtheta_dt + dtheta_dt_pred)
            omega[i + 1] = omega[i] + (h / 2.0) * (domega_dt + domega_dt_pred)

        return theta, omega


def analytical_solution(pendulum, time):
    """
    计算任意角度简谐摆的解析解（使用Jacobi椭圆函数）
    """
    k = np.sin(pendulum.theta0 / 2)  # 模数
    omega0 = np.sqrt(pendulum.g / pendulum.L)  # 自然频率

    # 使用Jacobi椭圆函数
    sn, _, _, _ = ellipj(omega0 * time + np.pi / 2, k * k)
    return 2 * np.arcsin(k * sn)


def analytical_omega(pendulum, time):
    """
    计算解析解的角速度
    """
    k = np.sin(pendulum.theta0 / 2)
    omega0 = np.sqrt(pendulum.g / pendulum.L)
    sn, cn, _, _ = ellipj(omega0 * time + np.pi / 2, k * k)
    return 2 * k * omega0 * cn / np.sqrt(1 - k * k * sn * sn)


def compute_error(theta_numeric, theta_analytic):
    """
    计算数值方法与解析解的误差。

    参数:
    - theta_numeric (np.ndarray): 数值方法计算得到的角度数组（rad）。
    - theta_analytic (np.ndarray): 解析解角度数组（rad）。

    返回:
    - error (np.ndarray): 误差数组（rad）。
    """
    return np.abs(theta_numeric - theta_analytic)


def get_user_input(prompt, default):
    """
    获取用户输入，如果用户不输入则返回默认值。

    参数:
    - prompt (str): 提示信息。
    - default (float): 默认值。

    返回:
    - value (float): 用户输入的值或默认值。
    """
    try:
        user_input = input(f"{prompt} (默认: {default}): ")
        return float(user_input) if user_input else default
    except ValueError:
        print("Invalid input, using default value.")
        return default


def main():
    print("请输入摆的参数(直接回车使用默认值):")
    length = get_user_input("摆长 L (m)", 1.0)
    mass = get_user_input("质量 m (kg)", 1.0)
    gravity = get_user_input("重力加速度 g (m/s²)", 9.81)
    theta0 = get_user_input("初始角度 θ₀ (rad)", 1.0)
    omega0 = get_user_input("初始角速度 ω₀ (rad/s)", 0.0)

    pendulum = Pendulum(length, mass, gravity, theta0, omega0)

    h = get_user_input("时间步长 h (s)", 0.05)
    T = get_user_input("总模拟时间 T (s)", 50.0)
    N = int(T / h)
    time = np.linspace(0, T, N + 1)

    methods = {
        "Euler": pendulum.euler,
        "Midpoint": pendulum.midpoint,
        "RK4": pendulum.rk4,
        "Euler-Trapezoidal": pendulum.euler_trapezoidal,
    }

    results = {}
    omegas = {}
    energies = {}
    errors = {}

    theta_analytic = analytical_solution(pendulum, time)
    omega_analytic = analytical_omega(pendulum, time)
    E0 = (
        pendulum.m * pendulum.g * pendulum.L * (1 - np.cos(pendulum.theta0))
        + 0.5 * pendulum.m * (pendulum.L * pendulum.omega0) ** 2
    )
    energy_analytic = np.full_like(time, E0)

    for name, method in methods.items():
        theta, omega = method(h, N)
        energy = pendulum.compute_energy(theta, omega)
        error = compute_error(theta, theta_analytic)
        results[name] = theta
        omegas[name] = omega
        energies[name] = energy
        errors[name] = error

    # 创建4个图表
    fig1, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, figsize=(10, 8))
    fig2, ((ax21, ax22), (ax23, ax24)) = plt.subplots(2, 2, figsize=(10, 8))
    fig3, ((ax31, ax32), (ax33, ax34)) = plt.subplots(2, 2, figsize=(10, 8))
    fig4, ((ax41, ax42), (ax43, ax44)) = plt.subplots(2, 2, figsize=(10, 8))

    axes1 = [ax11, ax12, ax13, ax14]  # 角度图
    axes2 = [ax21, ax22, ax23, ax24]  # 角速度图
    axes3 = [ax31, ax32, ax33, ax34]  # 能量图
    axes4 = [ax41, ax42, ax43, ax44]  # 误差图

    for (name, theta), ax1, ax2, ax3, ax4 in zip(
        results.items(), axes1, axes2, axes3, axes4
    ):
        # 角度图
        ax1.plot(time, theta, label=name, color="blue", linewidth=1)
        ax1.plot(
            time,
            theta_analytic,
            label="Analytical",
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Angle (rad)")
        ax1.set_title(f"{name} - Angle vs Time")
        ax1.legend()
        ax1.grid(True)

        # 角速度图
        ax2.plot(time, omegas[name], label=name, color="blue", linewidth=1)
        ax2.plot(
            time,
            omega_analytic,
            label="Analytical",
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Angular Velocity (rad/s)")
        ax2.set_title(f"{name} - Angular Velocity vs Time")
        ax2.legend()
        ax2.grid(True)

        # 能量图
        ax3.plot(time, energies[name], label=name, color="blue", linewidth=1)
        ax3.plot(
            time,
            energy_analytic,
            label="Analytical",
            color="red",
            linestyle="--",
            linewidth=1,
        )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Energy (J)")
        ax3.set_title(f"{name} - Energy vs Time")
        ax3.legend()
        ax3.grid(True)

        # 误差图
        ax4.plot(time, errors[name], label=name, color="blue", linewidth=1)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Error (rad)")
        ax4.set_title(f"{name} - Error vs Analytical")
        ax4.legend()
        ax4.grid(True)

    for fig in [fig1, fig2, fig3, fig4]:
        fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
