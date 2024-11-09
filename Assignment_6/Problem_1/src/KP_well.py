import numpy as np
import time
from scipy import constants as const
from scipy.sparse import diags, coo_matrix
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
    """生成傅里叶级数的自然顺序列表：[0,1,...N,-N,...,-1]"""
    return np.concatenate((np.arange(0, N + 1), np.arange(-N, 0)))


def compute_Vq_analytical(N, U0, Lw, Lb, a):
    """解析计算势能傅里叶系数，使用自然顺序"""
    n_list = get_n_list(N)
    Vq = np.zeros(2 * N + 1, dtype=complex)
    non_zero = n_list != 0
    Vq[non_zero] = (U0 / (2 * np.pi * n_list[non_zero] * 1j)) * (
        np.exp(-2j * np.pi * n_list[non_zero] * Lw / a) - 1
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
    return diags(kinetic, offsets=0, format="csr")  # 'csr'表示压缩稀疏行格式


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


def solve_eigenvalues(H_matrix, num_levels=3):
    """求解本征值和本征函数，添加异常处理，并排序"""
    try:
        eigenvalues, eigenvectors = eigsh(H_matrix, k=num_levels, which="SA")
        # 确保排序
        sorted_indices = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        return eigenvalues, eigenvectors
    except Exception as e:
        print(f"Error in eigenvalue solver: {e}")
        return None, None


def check_degeneracy(eigenvalues, tolerance=1e-4, relative=True):
    """
    检查能级简并性
    返回是否存在简并的布尔值
    参数:
        relative: 是否使用相对误差比较
    """
    sorted_eigenvalues = np.sort(eigenvalues)
    for i in range(1, len(sorted_eigenvalues)):
        if relative:
            if sorted_eigenvalues[i] != 0 and abs(
                sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]
            ) < tolerance * abs(sorted_eigenvalues[i]):
                return True
            elif (
                sorted_eigenvalues[i] == 0
                and abs(sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]) < tolerance
            ):
                return True
        else:
            if abs(sorted_eigenvalues[i] - sorted_eigenvalues[i - 1]) < tolerance:
                return True
    return False


def reconstruct_wavefunction_vectorized(eigenvectors, q_values, a):
    """
    利用本征矢量重构波函数，考虑周期性，将范围扩展到 -a 到 2a，便于绘图
    归一化基于本征向量的模长
    """
    x = np.linspace(0, a, 1000)  # 仅在一个周期内计算
    exponentials = np.exp(1j * 2 * np.pi / a * np.outer(q_values, x))
    wavefunctions = []
    num_levels = eigenvectors.shape[1]  # 获取本征向量的列数，即能级数量
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


def plot_energy_levels(
    eigenvalues_eV, N, num_levels=3, method_label="Method A", fit=False, k_fit=0
):
    """
    绘制能级图，能级编号为x轴，能量为y轴，不同能级使用不同颜色
    如果fit为True，则绘制拟合曲线
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
    plt.xlabel("Energy Level (n)")
    plt.ylabel("Energy (eV)")
    plt.title(f"Energy Levels (N={N}, {method_label})")
    plt.xticks(levels)  # 设置x轴刻度为整数
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_wavefunctions_and_potential(x_values, wavefunctions, V_x):
    """
    绘制波函数的概率密度和势能分布
    """
    _, ax1 = plt.subplots(figsize=(12, 8))

    num_levels = len(wavefunctions)  # 获取波函数的数量
    colors = plt.cm.viridis(np.linspace(0, 1, num_levels))
    for i in range(num_levels):
        ax1.plot(
            x_values * 1e9,  # 将位置从米转换为纳米
            np.abs(wavefunctions[i]) ** 2 * 1e-9,  # 转换单位为1/nm
            color=colors[i],
            label=f"Level {i+1}",
        )
    ax1.set_xlabel("Position (nm)")
    ax1.set_ylabel(r"$|\Psi(x)|^2$ (1/nm)", color="tab:red")
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
    检查能级的收敛性，仅用于小规模测试
    """
    previous_eigenvalues = None
    for N_fourier in range(10, max_N + 1, 10):
        q_values = get_n_list(N_fourier)
        V_fourier = compute_Vq_analytical(N_fourier, U0, Lw, Lb, a)
        T_matrix = build_kinetic_matrix(q_values, a, hbar, m_e)
        V_matrix = build_potential_matrix_optimized(N_fourier, V_fourier)
        H_matrix = T_matrix + V_matrix
        eigenvalues, eigenvectors = solve_eigenvalues(H_matrix, num_levels=num_levels)
        if eigenvalues is None or eigenvectors is None:
            print(f"Skipping N={N_fourier} due to solver error.")
            continue
        eigenvalues_eV = eigenvalues / e
        print(f"N={N_fourier}, Lowest {num_levels} eigenvalues (eV): {eigenvalues_eV}")
        degeneracies = check_degeneracy(
            eigenvalues_eV, tolerance=tolerance, relative=True
        )
        if degeneracies:
            print("Degeneracies found.")
        else:
            print("No degeneracies found.")
        if previous_eigenvalues is not None:
            # 使用相对误差比较
            relative_diff = np.abs(eigenvalues_eV - previous_eigenvalues) / np.abs(
                previous_eigenvalues
            )
            if np.all(relative_diff < tolerance):
                print(f"Converged at N={N_fourier}\n")
                break
        previous_eigenvalues = eigenvalues_eV
    else:
        print("Did not converge within the maximum N range.")


def compare_fourier_methods(N_fourier, U0, Lw, Lb, a):
    """
    比较解析方法和FFT方法计算傅里叶系数的结果，使用绝对误差
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

    # 结果对比：计算绝对误差
    abs_diff = np.abs(Vq_analytical - Vq_fft)
    max_abs_diff = np.max(abs_diff) / e  # 转换为 eV
    mean_abs_diff = np.mean(abs_diff) / e  # 转换为 eV

    print("\n傅里叶系数计算方法比较：")
    print(f"傅里叶级数范围：[-{N_fourier}, {N_fourier}]")
    print(f"解析方法用时：{time_analytical:.2e}秒")
    print(f"FFT方法用时：{time_fft:.2e}秒")
    print(f"最大绝对差异：{max_abs_diff:.6e} eV")
    print(f"平均绝对差异：{mean_abs_diff:.6e} eV")

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
    plt.plot(n_values, abs_diff / e, "k-")
    plt.xlabel("n")
    plt.ylabel("绝对差异 (eV)")
    plt.title("解析方法与FFT方法的绝对差异")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return Vq_analytical, Vq_fft


def build_and_solve(N, num_levels, U0, Lw, Lb, a, hbar, m_e):
    """构建矩阵，求解本征值和本征向量，返回 sorted eigenvalues and eigenvectors"""
    q_values = get_n_list(N)
    V_fourier = compute_Vq_analytical(N, U0, Lw, Lb, a)
    V_matrix = build_potential_matrix_optimized(N, V_fourier)
    T_matrix = build_kinetic_matrix(q_values, a, hbar, m_e)
    H_matrix = T_matrix + V_matrix
    eigenvalues, eigenvectors = solve_eigenvalues(H_matrix, num_levels)
    if eigenvalues is None or eigenvectors is None:
        return None, None
    eigenvalues_eV = eigenvalues / e
    return eigenvalues_eV, eigenvectors


def print_energy_levels(eigenvalues_eV, num_levels=3):
    """打印前 num_levels 个能级，格式为 .4e"""
    print(f"Lowest {num_levels} eigenvalues (eV):")
    for idx, energy in enumerate(eigenvalues_eV[:num_levels], start=1):
        print(f"  Level {idx}: {energy:.4e} eV")


def main(
    U0_eV=2.0,
    Lw_nm=0.9,
    Lb_nm=0.1,
    N_initial=50,
    num_levels_initial=5,
    medium_N=100,
    medium_num_levels=20,
    large_N=500,
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

    # 步骤2：进行收敛性检查，仅对小规模N进行
    print("第二步：进行收敛性检查（仅对小规模N进行）")
    check_convergence(
        U0_J, Lw, Lb, a, hbar, m_e, num_levels=3, max_N=50, tolerance=1e-4
    )

    # 步骤3：使用解析方法计算的傅里叶系数进行AB方法对比
    print("\n第三步：使用解析方法计算的傅里叶系数进行AB方法对比")
    # 方法A：Vq' ∈ [-N, N]
    print(f"\n方法A（Vq' ∈ [-{N_initial}, {N_initial}]）")
    eigenvalues_A_eV, eigenvectors_A = build_and_solve(
        N_initial, num_levels_initial, U0_J, Lw, Lb, a, hbar, m_e
    )
    if eigenvalues_A_eV is None or eigenvectors_A is None:
        print("方法A求解失败，跳过方法B对比。\n")
        return
    print_energy_levels(eigenvalues_A_eV, num_levels=3)
    degeneracies_A = check_degeneracy(
        eigenvalues_A_eV[:3], tolerance=1e-2, relative=True  # 放宽容差到1e-2
    )
    if degeneracies_A:
        print("Degeneracies found.\n")
    else:
        print("No degeneracies found.\n")

    # 方法B：Vq' ∈ [-2N, 2N]
    print(f"方法B（Vq' ∈ [-{2*N_initial}, {2*N_initial}]）")
    eigenvalues_B_eV, eigenvectors_B = build_and_solve(
        2 * N_initial, num_levels_initial, U0_J, Lw, Lb, a, hbar, m_e
    )
    if eigenvalues_B_eV is None or eigenvectors_B is None:
        print("方法B求解失败。\n")
        return
    print_energy_levels(eigenvalues_B_eV, num_levels=3)
    degeneracies_B = check_degeneracy(
        eigenvalues_B_eV[:3], tolerance=1e-2, relative=True  # 放宽容差到1e-2
    )
    if degeneracies_B:
        print("Degeneracies found.\n")
    else:
        print("No degeneracies found.\n")

    # 检查AB方法的差异
    energy_diff_AB = np.abs(eigenvalues_A_eV[:3] - eigenvalues_B_eV[:3]) / np.abs(
        eigenvalues_A_eV[:3]
    )
    print(f"方法A和方法B的前3个能级相对差异：{energy_diff_AB}\n")
    if np.all(energy_diff_AB < 1e-4):
        print("方法A和方法B的能级差异非常小。\n")
    else:
        print("方法A和方法B的能级差异存在显著差异。\n")

    # 步骤4：使用中等规模N进行绘制，然后询问是否进行大规模N的绘制与拟合
    print(f"\n第四步：使用中等规模N={medium_N}进行能级绘制")
    # 中等规模N
    eigenvalues_medium_eV, eigenvectors_medium = build_and_solve(
        medium_N, medium_num_levels, U0_J, Lw, Lb, a, hbar, m_e
    )
    if eigenvalues_medium_eV is None or eigenvectors_medium is None:
        print("中等规模N求解失败，无法进行绘制。\n")
    else:
        # 输出所有能级信息，使用.4e的精度
        print(f"N={medium_N}, Lowest {medium_num_levels} eigenvalues (eV):")
        for idx, energy in enumerate(
            eigenvalues_medium_eV[:medium_num_levels], start=1
        ):
            print(f"  Level {idx}: {energy:.4e} eV")
        degeneracies_medium = check_degeneracy(
            eigenvalues_medium_eV[:medium_num_levels],
            tolerance=1e-2,
            relative=True,  # 放宽容差到1e-2
        )
        if degeneracies_medium:
            print("Degeneracies found.\n")
        else:
            print("No degeneracies found.\n")

        # 重构前三个波函数
        x_medium, wavefunctions_medium = reconstruct_wavefunction_vectorized(
            eigenvectors_medium[:, :3], get_n_list(medium_N), a
        )

        # 生成势阱
        V_x_medium = generate_potential_vectorized(x_medium, a, Lw, U0_J)

        # 绘制前三个波函数和势阱
        plot_wavefunctions_and_potential(x_medium, wavefunctions_medium, V_x_medium)

        # 绘制能级图
        plot_energy_levels(
            eigenvalues_medium_eV,
            medium_N,
            num_levels=medium_num_levels,
            method_label="Medium N",
        )

    # 询问用户是否进行大规模N的绘制与拟合
    proceed_large = (
        input(f"\n是否进行大规模N={large_N}的能级绘制与拟合？（y/n）：").strip().lower()
    )
    if proceed_large != "y":
        print("跳过大规模N的能级绘制与拟合。")
        return

    # 进行最大规模N的绘制与拟合
    print("\n继续进行大规模N的能级绘制与拟合")
    eigenvalues_large_eV, eigenvectors_large = build_and_solve(
        large_N, large_num_levels, U0_J, Lw, Lb, a, hbar, m_e
    )
    if eigenvalues_large_eV is None or eigenvectors_large is None:
        print("大规模N求解失败，无法进行绘制。\n")
    else:
        # 输出所有能级信息，使用.4e的精度
        print(f"N={large_N}, Lowest {large_num_levels} eigenvalues (eV):")
        for idx, energy in enumerate(eigenvalues_large_eV[:large_num_levels], start=1):
            print(f"  Level {idx}: {energy:.4e} eV")
        degeneracies_large = check_degeneracy(
            eigenvalues_large_eV[:large_num_levels],
            tolerance=1e-2,
            relative=True,  # 放宽容差到1e-2
        )
        if degeneracies_large:
            print("Degeneracies found.\n")
        else:
            print("No degeneracies found.\n")

        # 绘制能级图并加入拟合曲线
        levels_large = np.arange(1, large_num_levels + 1)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            levels_large,
            eigenvalues_large_eV,
            c=levels_large,
            cmap="viridis",
            s=10,
            label="Computed Energy Levels",
        )
        plt.xlabel("Energy Level (n)")
        plt.ylabel("Energy (eV)")
        plt.title(f"Energy Levels for N={large_N} (Method A)")
        plt.colorbar(label="Energy Level")

        # 拟合
        popt, pcov = curve_fit(quadratic_fit, levels_large, eigenvalues_large_eV)
        k_fit = popt[0]
        E_n_fit = quadratic_fit(levels_large, k_fit)

        plt.plot(
            levels_large, E_n_fit, "r--", label=f"$k \\cdot n^2$ Fit, k={k_fit:.4f} eV"
        )
        # 设置x轴刻度为整数点，减少密度
        plt.xticks(np.concatenate(([1], np.arange(10, large_num_levels + 1, step=10))))
        plt.legend()
        plt.grid(True)
        plt.show()

        # 输出拟合结果
        print(f"拟合结果：k = {k_fit:.6f} eV")


if __name__ == "__main__":
    main()
