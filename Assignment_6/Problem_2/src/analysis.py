import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取数据
months, sunspots = np.loadtxt("sunspots.txt", unpack=True)
print(f"月份数: {len(months)}, 太阳黑子数: {len(sunspots)}")
print("前5条数据:", months[:5], sunspots[:5])
N = len(sunspots)


# 定义FFT和功率谱计算函数
def compute_fft(signal):
    fft_coefficients = np.fft.fft(signal)
    power_spectrum = np.abs(fft_coefficients) ** 2
    k_values = np.fft.fftfreq(N, d=1 / N)
    positive_indices = np.where(k_values >= 0)
    k_positive = k_values[positive_indices]
    power_positive = power_spectrum[positive_indices]
    return k_positive, power_positive, fft_coefficients


# 2. 去趋势和窗口化处理
sunspots_detrended = detrend(sunspots)  # 去趋势
window = np.hanning(len(sunspots_detrended))  # 生成汉宁窗
sunspots_detrended_windowed = sunspots_detrended * window  # 叠加去趋势和窗口化

# 3. 计算不同处理方式下的FFT和功率谱
k_original, power_original, fft_original = compute_fft(sunspots)
k_detrended, power_detrended, fft_detrended = compute_fft(sunspots_detrended)
k_windowed, power_windowed, fft_windowed = compute_fft(sunspots * window)  # 仅窗口化
k_detrended_windowed, power_detrended_windowed, fft_detrended_windowed = compute_fft(
    sunspots_detrended_windowed
)  # 去趋势 + 汉宁窗
# 打印前10个频率分量和对应的功率
print("原始数据前10个频率分量和对应的功率:")
for i in range(10):
    print(f"k={k_original[i]:.4f}, |c_k|^2={power_original[i]:.2f}")
print("\n去趋势后前10个频率分量和对应的功率:")
for i in range(10):
    print(f"k={k_detrended[i]:.4f}, |c_k|^2={power_detrended[i]:.2f}")
print("\n窗口化后前10个频率分量和对应的功率:")
for i in range(10):
    print(f"k={k_windowed[i]:.4f}, |c_k|^2={power_windowed[i]:.2f}")

# 4. 绘制功率谱对比图
plt.figure(figsize=(12, 8))
plt.plot(k_original[5:], power_original[5:], label="原始数据")
plt.plot(k_detrended[3:], power_detrended[3:], label="去趋势后")
plt.plot(k_windowed[3:], power_windowed[3:], label="施加汉宁窗后")
plt.plot(
    k_detrended_windowed[1:], power_detrended_windowed[1:], label="去趋势 + 汉宁窗后"
)

plt.xlabel(r"$k$ (频率)")
plt.ylabel(r"$|c_k|^2$ (功率谱)")
plt.title("不同处理方式下的太阳黑子功率谱对比")
plt.legend()
plt.grid(True)
plt.xlim(0, N // 20)
plt.show()

# 5. 峰值分析
peak_k_index = np.argmax(power_detrended_windowed[1:]) + 1
peak_k = k_detrended_windowed[peak_k_index]
peak_power = power_detrended_windowed[peak_k_index]
frequency = peak_k
period = N / frequency
print(f"峰值对应的 k 值为: {peak_k:.4f}")
print(f"对应的周期为: {period:.2f} 月")

# 6. 重构周期性信号
fft_filtered = np.zeros_like(fft_detrended_windowed)
fft_filtered[peak_k_index] = fft_detrended_windowed[peak_k_index]
fft_filtered[-peak_k_index] = fft_detrended_windowed[-peak_k_index]
periodic_signal = np.fft.ifft(fft_filtered).real

# 7. 原始与重构信号对比
plt.figure(figsize=(14, 7))
plt.plot(months, sunspots, label="原始数据", alpha=0.7)
plt.plot(
    months,
    periodic_signal,
    label=f"重构周期性信号 (k={peak_k:.4f}, 周期={period:.2f} 月)",
    linestyle="--",
    color="red",
)
plt.xlabel("月份 since 1749年1月")
plt.ylabel("太阳黑子数")
plt.title("原始太阳黑子数据与重构周期性信号对比")
plt.legend()
plt.grid(True)
plt.show()

# 8. 绘制处理后的信号对比
plt.figure(figsize=(14, 7))
plt.plot(months, sunspots, label="原始数据", alpha=0.5)
plt.plot(months, sunspots_detrended, label="去趋势后", alpha=0.7)
plt.plot(months, sunspots * window, label="施加汉宁窗后", alpha=0.7)  # 窗口化
plt.plot(
    months, sunspots_detrended_windowed, label="去趋势 + 汉宁窗后", alpha=0.7
)  # 去趋势+窗口化
plt.xlabel("月份 since 1749年1月")
plt.ylabel("太阳黑子数")
plt.title("不同处理方式下的太阳黑子数据对比")
plt.legend()
plt.grid(True)
plt.show()
