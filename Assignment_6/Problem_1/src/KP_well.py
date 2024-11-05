import numpy as np
import time
from scipy import constants as const
from scipy.sparse import diags, csr_matrix, coo_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.optimize import curve_fit

# 物理常数定义
hbar = const.hbar  # J·s
m_e = const.m_e  # kg
e = const.e  # C
nm = 1e-9  # nm to m

# 绘图中文字体设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def get_n_list(N):
    """生成傅里叶级数的Python自然顺序列表：[0,1,...N,-N,...,-1]"""
    return np.concatenate((np.arange(0, N + 1), np.arange(-N, 0)))


def compute_Vq_analytical(N, U0, Lw, Lb, a):
    """解析计算势能傅里叶系数，使用自然顺序"""
    n_list = get_n_list(N)
    Vq = np.zeros(2 * N + 1, dtype=complex)
    non_zero = n_list != 0
    Vq[non_zero] = (U0 / (2 * np.pi * n_list[non_zero] * 1j)) * (
        np.exp(-2j * np.pi * n_list[non_zero] * Lw / a)
        - np.exp(-2j * np.pi * n_list[non_zero] * (Lw + Lb) / a)
    )
    Vq[0] = U0 * Lb / a
    return Vq


def compute_Vq_fft(N, U0, Lw, Lb, a):
    """使用FFT计算傅里叶系数，使用自然顺序"""
    N_samples = 2 * N + 1
    x = np.linspace(0, a, N_samples, endpoint=False)
    V_x = np.where((x >= Lw) & (x < a), U0, 0)
    return fft(V_x) / N_samples  # scipy的fft默认使用自然顺序


def build_kinetic_matrix(q_values, a, hbar, m_e):
    """构建动能矩阵"""
    kinetic = (hbar**2 * (2 * np.pi * q_values / a) ** 2) / (2 * m_e)  # J
    return diags(kinetic, offsets=0, format="csr")  #'csr'表示压缩稀疏行格式


def build_potential_matrix_optimized(N, V_fourier):
    """
    使用向量化和稀疏矩阵构建势能矩阵
    """
    H_size = 2 * N + 1
    Fourier_size = (len(V_fourier) - 1) // 2
    q_list = get_n_list(N)

    p = np.array(q_list)
    q = np.array(q_list)
    diff_matrix = p[:, None] - q[None, :]  # 构建差值矩阵

    # 创建掩码，选择 |diff| <= Fourier_size
    mask = np.abs(diff_matrix) <= Fourier_size

    # 过滤diff值，并映射到V_fourier的索引
    diff_filtered = diff_matrix[mask]
    V_values = V_fourier[diff_filtered]

    # 获取行和列索引
    row_indices, col_indices = np.where(mask)

    # 构建稀疏矩阵
    V_matrix = coo_matrix(
        (V_values, (row_indices, col_indices)), shape=(H_size, H_size)
    ).tocsr()

    return V_matrix


def solve_eigenvalues_sparse(H_matrix, num_levels=3):
    """求解本征值和本征函数，添加异常处理"""
    try:
        eigenvalues, eigenvectors = eigsh(H_matrix, k=num_levels, which="SA")
        return eigenvalues, eigenvectors
    except Exception as e:
        print(f"Error in eigenvalue solver: {e}")
        return None, None


def check_degeneracy(eigenvalues, tolerance=1e-5, relative=True):
    """
    检查能级简并性
    返回简并能级及其简并度的字典
    注：简并度从1开始计数，连续的能级差小于tolerance视为简并
    参数:
        relative: 是否使用相对误差比较
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    degeneracies = {}
    current_group = [sorted_eigenvalues[0]]

    for i in range(1, len(sorted_eigenvalues)):
        if relative:
            if sorted_eigenvalues[i] != 0 and abs(
                sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]
            ) < tolerance * abs(sorted_eigenvalues[i]):
                current_group.append(sorted_eigenvalues[i])
            elif (
                sorted_eigenvalues[i] == 0
                and abs(sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]) < tolerance
            ):
                current_group.append(sorted_eigenvalues[i])
            else:
                if len(current_group) > 1:
                    degeneracies[current_group[0]] = len(current_group)
                current_group = [sorted_eigenvalues[i]]
        else:
            if abs(sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]) < tolerance:
                current_group.append(sorted_eigenvalues[i])
            else:
                if len(current_group) > 1:
                    degeneracies[current_group[0]] = len(current_group)
                current_group = [sorted_eigenvalues[i]]

    if len(current_group) > 1:
        degeneracies[current_group[0]] = len(current_group)

    return degeneracies


def reconstruct_wavefunction_vectorized(eigenvectors, q_values, a, num_levels=3):
    """
    利用本征矢量重构波函数，考虑周期性，将范围扩展到 -a 到 2a，便于绘图
    归一化基于本征向量的模长
    """
    x = np.linspace(0, a, 1000)  # 仅在一个周期内计算
    exponentials = np.exp(1j * 2 * np.pi / a * np.outer(q_values, x))
    wavefunctions = []
    for i in range(num_levels):
        # eigenvectors[:, i] 是第i个本征向量，其元素对应各基底的系数
        # psi = sum(c_n * exp(i * 2pi n x / a)) / sqrt(a)
        psi = eigenvectors[:, i].dot(exponentials) / np.sqrt(a)
        # 计算本征向量的L2模长
        norm = np.linalg.norm(eigenvectors[:, i])
        psi_normalized = psi / norm
        wavefunctions.append(psi_normalized)

    # 扩展 x 和 wavefunctions 到 -a 到 2a
    x_extended = np.concatenate((x - a, x, x + a))
    # 由于拓展到3个周期，归一化除以 sqrt(3)
    wavefunctions_extended = [np.tile(psi, 3) / np.sqrt(3) for psi in wavefunctions]

    return x_extended, wavefunctions_extended


def generate_potential_vectorized(x, a, Lw, U0_J):
    """
    准备绘制原势阱
    """
    pos_in_cycle = np.mod(x, a)
    V_x = np.where((pos_in_cycle >= Lw) & (pos_in_cycle < a), U0_J, 0)
    return V_x


def plot_energy_levels_swapped(eigenvalues_eV, num_levels=3):
    """
    绘制能级图，能级编号为x轴，能量为y轴，不同能级使用不同颜色
    """
    levels = np.arange(1, num_levels + 1)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, num_levels))
    for i in range(num_levels):
        plt.scatter(levels[i], eigenvalues_eV[i], color=colors[i], label=f"Level {i+1}")
        plt.plot(
            [levels[i], levels[i]],
            [0, eigenvalues_eV[i]],
            color=colors[i],
            linestyle="--",
            alpha=0.5,
        )
    plt.xlabel("Energy Level")
    plt.ylabel("Energy (eV)")
    plt.title("Energy Levels (Method A)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_wavefunctions_and_potential(x_values, wavefunctions, V_x, num_levels=3):
    """
    绘制波函数的概率密度和势能分布
    """
    fig, ax1 = plt.subplots(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, num_levels))
    for i in range(num_levels):
        ax1.plot(
            x_values * 1e9,  # 将位置从米转换为纳米
            np.abs(wavefunctions[i]) ** 2,
            color=colors[i],
            label=f"Level {i+1}",
        )
    ax1.set_xlabel("Position (nm)")
    ax1.set_ylabel(r"$|\Psi(x)|^2$ (1/m)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # 绘制势阱
    ax2 = ax1.twinx()  # 共享x轴
    ax2.plot(
        x_values * 1e9, V_x / e, color="tab:blue", label="Potential V(x)", linewidth=2
    )
    ax2.set_ylabel("Potential V(x) (eV)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # 添加图例
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper right")

    plt.title(r"Wavefunctions $|\Psi(x)|^2$ and Potential V(x)")
    plt.grid(True)
    plt.show()


def quadratic_fit(n, k):
    """拟合函数：E = k * n^2"""
    return k * n**2


def check_convergence(U0, Lw, Lb, a, hbar, m_e, num_levels=3, max_N=50, tolerance=1e-4):
    """
    检查能级的收敛性
    """
    previous_eigenvalues = None
    for N_fourier in range(10, max_N + 1, 10):
        q_values = get_n_list(N_fourier)
        V_fourier = compute_Vq_analytical(N_fourier, U0, Lw, Lb, a)
        T_matrix = build_kinetic_matrix(q_values, a, hbar, m_e)
        V_matrix = build_potential_matrix_optimized(N_fourier, V_fourier)
        H_matrix = T_matrix + V_matrix
        eigenvalues, eigenvectors = solve_eigenvalues_sparse(
            H_matrix, num_levels=num_levels
        )
        if eigenvalues is None or eigenvectors is None:
            print(f"Skipping N_fourier={N_fourier} due to solver error.")
            continue
        # 确保排序
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        eigenvalues_eV = eigenvalues / e
        print(
            f"N_fourier={N_fourier}, Lowest {num_levels} eigenvalues (eV): {eigenvalues_eV}"
        )
        degeneracies = check_degeneracy(
            eigenvalues_eV, tolerance=tolerance, relative=True
        )
        if degeneracies:
            print(f"Degeneracies found: {degeneracies}")
        if previous_eigenvalues is not None:
            # 使用相对误差比较
            relative_diff = np.abs(eigenvalues_eV - previous_eigenvalues) / np.abs(
                previous_eigenvalues
            )
            if np.all(relative_diff < tolerance):
                print(f"Converged at N_fourier={N_fourier}\n")
                break
        previous_eigenvalues = eigenvalues_eV
    else:
        print("Did not converge within the maximum N range.")


def compare_fourier_methods(N_fourier, U0, Lw, Lb, a):
    """
    比较解析方法和FFT方法计算傅里叶系数的结果，使用相对误差
    """
    n_values = get_n_list(N_fourier)  # 生成自然顺序列表

    # 解析计算Vq
    start_time = time.time()
    Vq_analytical = compute_Vq_analytical(N_fourier, U0, Lw, Lb, a)
    time_analytical = time.time() - start_time

    # FFT计算Vq
    start_time = time.time()
    Vq_fft = compute_Vq_fft(N_fourier, U0, Lw, Lb, a)
    time_fft = time.time() - start_time

    # 结果对比：计算相对误差
    # 避免除以零，使用掩码
    mask = Vq_analytical != 0
    relative_diff = np.zeros_like(Vq_analytical, dtype=float)
    relative_diff[mask] = np.abs(Vq_analytical[mask] - Vq_fft[mask]) / np.abs(
        Vq_analytical[mask]
    )
    relative_diff[~mask] = np.abs(Vq_fft[~mask])  # 当Vq_analytical为0时，使用绝对误差

    max_rel_diff = np.max(relative_diff)
    mean_rel_diff = np.mean(relative_diff)

    print("\n傅里叶系数计算方法比较：")
    print(f"傅里叶级数范围：[-{N_fourier}, {N_fourier}]")
    print(f"解析方法用时：{time_analytical:.6f}秒")
    print(f"FFT方法用时：{time_fft:.6f}秒")
    print(f"最大相对差异：{max_rel_diff:.6e}")
    print(f"平均相对差异：{mean_rel_diff:.6e}")

    # 绘制对比图
    plt.figure(figsize=(12, 8))

    plt.subplot(211)
    plt.plot(n_values, np.abs(Vq_analytical) / e, "r-", label="解析方法")
    plt.plot(n_values, np.abs(Vq_fft) / e, "b--", label="FFT方法")
    plt.xlabel("n")
    plt.ylabel("|Vq|/eV")
    plt.title("傅里叶系数幅值对比")
    plt.legend()
    plt.grid(True)

    plt.subplot(212)
    plt.plot(n_values, relative_diff, "k-")
    plt.xlabel("n")
    plt.ylabel("相对差异")
    plt.title("解析方法与FFT方法的相对差异")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return Vq_analytical, Vq_fft


def main(
    U0_eV=2.0,
    Lw_nm=0.9,
    Lb_nm=0.1,
    N_initial=50,
    num_levels_initial=5,
    large_N=200,
    large_num_levels=100,
):
    # 单位转换
    U0_J = U0_eV * e  # J
    a_nm = Lw_nm + Lb_nm  # nm
    a = a_nm * nm  # m
    Lw = Lw_nm * nm  # m
    Lb = Lb_nm * nm  # m

    # 步骤1：比较解析方法和FFT方法计算傅里叶系数的差异
    print("第一步：比较解析方法和FFT方法计算傅里叶系数的差异")
    Vq_analytical, Vq_fft = compare_fourier_methods(N_initial, U0_J, Lw, Lb, a)

    # 检查差异是否很小
    if np.allclose(Vq_analytical, Vq_fft, atol=1e-6):
        print("解析方法和FFT方法计算的傅里叶系数差异非常小。\n")
    else:
        print("解析方法和FFT方法计算的傅里叶系数存在显著差异。\n")

    # 步骤2：进行收敛性检查
    print("第二步：进行收敛性检查")
    check_convergence(
        U0_J, Lw, Lb, a, hbar, m_e, num_levels=3, max_N=50, tolerance=1e-4
    )

    # 步骤3：使用FFT计算得到的系数进行较小N的AB方法对比
    print("第三步：使用FFT方法计算的傅里叶系数进行AB方法对比")
    # 方法A：Vq' ∈ [-N, N]
    start_time_A = time.time()
    q_values_A = get_n_list(N_initial)
    V_fourier_A = Vq_fft  # 使用FFT方法得到的傅里叶系数
    V_matrix_A = build_potential_matrix_optimized(N_initial, V_fourier_A)
    T_matrix_A = build_kinetic_matrix(q_values_A, a, hbar, m_e)
    H_matrix_A = T_matrix_A + V_matrix_A
    eigenvalues_A, eigenvectors_A = solve_eigenvalues_sparse(
        H_matrix_A, num_levels_initial
    )
    if eigenvalues_A is None or eigenvectors_A is None:
        print("方法A求解失败，跳过方法B对比。\n")
        return
    # 确保排序
    sorted_indices_A = np.argsort(eigenvalues_A)
    eigenvalues_A = eigenvalues_A[sorted_indices_A]
    eigenvectors_A = eigenvectors_A[:, sorted_indices_A]
    eigenvalues_eV_A = eigenvalues_A / e
    end_time_A = time.time()
    time_A = end_time_A - start_time_A

    print("方法A（Vq' ∈ [-N, N]）")
    print(f"最低三个能级（eV）：{eigenvalues_eV_A[:3]}")
    degeneracies_A = check_degeneracy(
        eigenvalues_eV_A[:3], tolerance=1e-5, relative=True
    )
    if degeneracies_A:
        print(f"Degeneracies found: {degeneracies_A}")
    else:
        print("No degeneracies found.")
    print(f"计算时间：{time_A:.4f} 秒\n")

    # 方法B：Vq' ∈ [-2N, 2N]
    print("方法B（Vq' ∈ [-2N, 2N]）")
    start_time_B = time.time()
    N_extended = 2 * N_initial
    V_fourier_extended = compute_Vq_analytical(N_extended, U0_J, Lw, Lb, a)
    V_matrix_B = build_potential_matrix_optimized(N_initial, V_fourier_extended)
    H_matrix_B = T_matrix_A + V_matrix_B  # 动能矩阵不变
    eigenvalues_B, eigenvectors_B = solve_eigenvalues_sparse(
        H_matrix_B, num_levels_initial
    )
    if eigenvalues_B is None or eigenvectors_B is None:
        print("方法B求解失败。\n")
        return
    # 确保排序
    sorted_indices_B = np.argsort(eigenvalues_B)
    eigenvalues_B = eigenvalues_B[sorted_indices_B]
    eigenvectors_B = eigenvectors_B[:, sorted_indices_B]
    eigenvalues_eV_B = eigenvalues_B / e
    end_time_B = time.time()
    time_B = end_time_B - start_time_B

    print("方法B（Vq' ∈ [-2N, 2N]）")
    print(f"最低三个能级（eV）：{eigenvalues_eV_B[:3]}")
    degeneracies_B = check_degeneracy(
        eigenvalues_eV_B[:3], tolerance=1e-5, relative=True
    )
    if degeneracies_B:
        print(f"Degeneracies found: {degeneracies_B}")
    else:
        print("No degeneracies found.")
    print(f"计算时间：{time_B:.4f} 秒\n")

    # 检查AB方法的差异
    energy_diff_AB = np.abs(eigenvalues_eV_A[:3] - eigenvalues_eV_B[:3]) / np.abs(
        eigenvalues_eV_A[:3]
    )
    print(f"方法A和方法B的前3个能级相对差异：{energy_diff_AB}\n")
    if np.all(energy_diff_AB < 1e-4):
        print("方法A和方法B的能级差异非常小。\n")
    else:
        print("方法A和方法B的能级差异存在显著差异。\n")

    # 步骤4：使用较大的N和num_levels绘制能级图，并用n^2拟合
    print("第四步：使用较大的N和num_levels绘制能级图，并用n^2拟合")
    large_N = 200
    large_num_levels = 100

    # 使用FFT方法计算傅里叶系数
    Vq_large = compute_Vq_fft(large_N, U0_J, Lw, Lb, a)
    q_values_large = get_n_list(large_N)
    V_matrix_large = build_potential_matrix_optimized(large_N, Vq_large)
    T_matrix_large = build_kinetic_matrix(q_values_large, a, hbar, m_e)
    H_matrix_large = T_matrix_large + V_matrix_large
    eigenvalues_large, eigenvectors_large = solve_eigenvalues_sparse(
        H_matrix_large, large_num_levels
    )
    if eigenvalues_large is None or eigenvectors_large is None:
        print("大规模求解失败，无法绘制能级图。\n")
        return
    # 确保排序
    sorted_indices_large = np.argsort(eigenvalues_large)
    eigenvalues_large = eigenvalues_large[sorted_indices_large]
    eigenvectors_large = eigenvectors_large[:, sorted_indices_large]
    eigenvalues_eV_large = eigenvalues_large / e

    # 重构波函数
    x_large, wavefunctions_large = reconstruct_wavefunction_vectorized(
        eigenvectors_large, q_values_large, a, large_num_levels
    )

    # 生成势阱
    V_x_large = generate_potential_vectorized(x_large, a, Lw, U0_J)

    # 绘制能级图
    levels_large = np.arange(1, large_num_levels + 1)
    plt.figure(figsize=(10, 6))
    plt.scatter(
        levels_large, eigenvalues_eV_large, c=levels_large, cmap="viridis", s=10
    )
    plt.xlabel("Energy Level")
    plt.ylabel("Energy (eV)")
    plt.title("Energy Levels for Large N (Method A)")
    plt.colorbar(label="Energy Level")
    plt.grid(True)
    plt.show()

    # 拟合能级与n^2的关系
    plt.figure(figsize=(10, 6))
    plt.plot(levels_large, eigenvalues_eV_large, "bo", label="Computed Energy Levels")

    # 拟合
    popt, pcov = curve_fit(quadratic_fit, levels_large, eigenvalues_eV_large)
    k_fit = popt[0]
    E_n_fit = quadratic_fit(levels_large, k_fit)

    plt.plot(
        levels_large, E_n_fit, "r--", label=f"$k \\cdot n^2$ Fit, k={k_fit:.4f} eV"
    )
    plt.xlabel("Energy Level")
    plt.ylabel("Energy (eV)")
    plt.title("Energy Levels vs $n^2$ Fit")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制波函数和势阱
    plot_wavefunctions_and_potential(
        x_large, wavefunctions_large, V_x_large, num_levels=3
    )


if __name__ == "__main__":
    main()
