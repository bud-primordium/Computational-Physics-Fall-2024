import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
zh_font = font_manager.FontProperties(fname='C:/Windows/Fonts/simsun.ttc')

# 定义总时间和时间步长
t_total = 10  # 总模拟时间
dt = 0.01     # 时间步长
n_steps = int(t_total / dt)  # 总步数

# 初始化时间数组
t = np.linspace(0, t_total, n_steps)

# 定义初始条件
x0 = 1.0  # 初始位置
v0 = 0.0  # 初始速度
omega = 1.0  # 角频率

# 定义存储位置和速度的数组
x_euler = np.zeros(n_steps)
v_euler = np.zeros(n_steps)

x_verlet = np.zeros(n_steps)
v_verlet = np.zeros(n_steps)

x_exact = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)  # 精确解

# 设置初始条件
x_euler[0] = x0
v_euler[0] = v0

x_verlet[0] = x0
v_verlet[0] = v0

# 欧拉算法模拟
for i in range(n_steps - 1):
    # 计算加速度 a = -ω^2 x
    a = -omega**2 * x_euler[i]
    # 更新位置和速度
    x_euler[i+1] = x_euler[i] + v_euler[i] * dt
    v_euler[i+1] = v_euler[i] + a * dt

# Verlet算法模拟
# 首先使用欧拉算法计算第二个位置点，以启动Verlet算法
x_verlet[1] = x_verlet[0] + v_verlet[0] * dt + 0.5 * (-omega**2 * x_verlet[0]) * dt**2

for i in range(1, n_steps - 1):
    # 计算加速度
    a = -omega**2 * x_verlet[i]
    # 更新位置
    x_verlet[i+1] = 2 * x_verlet[i] - x_verlet[i-1] + a * dt**2
    # 更新速度（可选，若需要计算速度，可用以下公式）
    v_verlet[i] = (x_verlet[i+1] - x_verlet[i-1]) / (2 * dt)
# 计算最后一个速度点
v_verlet[-1] = (x_verlet[-1] - x_verlet[-2]) / dt

# 计算误差
error_euler = np.abs(x_euler - x_exact)
error_verlet = np.abs(x_verlet - x_exact)

# 绘制误差曲线
plt.figure(figsize=(10, 6))
plt.plot(t, error_euler, label='欧拉算法误差')
plt.plot(t, error_verlet, label='Verlet算法误差')
plt.yscale('log')  # 使用对数坐标显示误差
plt.xlabel('时间 t', fontproperties=zh_font)
plt.ylabel('位置误差 |x - x_exact|', fontproperties=zh_font)
plt.title('欧拉算法与Verlet算法的全局误差比较', fontproperties=zh_font)
plt.legend(prop=zh_font)
plt.grid(True)
plt.show()