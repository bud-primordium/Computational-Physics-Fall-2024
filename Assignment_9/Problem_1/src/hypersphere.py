import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from tqdm import tqdm
import sys
import platform
import time


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


configure_matplotlib_fonts()


def exact_volume(n):
    """计算n维单位超球体的理论体积"""
    return (np.pi ** (n / 2)) / gamma(n / 2 + 1)


def monte_carlo_uniform(n, num_samples):
    """简单均匀采样蒙特卡洛方法"""
    points = np.random.uniform(-1, 1, (num_samples, n))
    distances_sq = np.sum(points**2, axis=1)
    count_inside = np.sum(distances_sq <= 1)
    return (2**n) * (count_inside / num_samples)


def importance_sampling_volume(n, num_samples, sigma=1.0):
    """重要性采样估计n维单位超球体的体积"""
    mean = np.zeros(n)
    cov = (sigma**2) * np.eye(n)
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    distances_sq = np.sum(samples**2, axis=1)
    inside = distances_sq <= 1.0
    f_x = inside.astype(float)
    q_x = (1 / ((2 * np.pi * sigma**2) ** (n / 2))) * np.exp(
        -distances_sq / (2 * sigma**2)
    )
    q_x[q_x == 0] = 1e-300  # 避免除以零
    weights = f_x / q_x
    return np.mean(weights)


def visualize_sampling(n, points, method):
    """根据维度n和采样方法可视化采样点"""
    if n > 3:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(points)
        plt.figure(figsize=(8, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=1)
        plt.title(f"{method} 生成的 {n} 维点的2D可视化")
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.grid(True)
        plt.axis("equal")
        plt.show()
    elif n == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.3,
            s=1,
            c="blue",
            label="Inside",
        )
        # 绘制单位球边界
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="green", linewidth=0.5, alpha=0.5)
        ax.set_title(f"{method} 生成的 3 维点的3D可视化")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()
        plt.show()
    elif n == 2:
        plt.figure(figsize=(8, 8))  # 设置为正方形
        # 分离内部和外部点
        inside = np.linalg.norm(points, axis=1) <= 1
        outside = ~inside
        plt.scatter(
            points[inside, 0],
            points[inside, 1],
            alpha=0.5,
            s=1,
            c="blue",
            label="Inside",
        )
        plt.scatter(
            points[outside, 0],
            points[outside, 1],
            alpha=0.2,
            s=1,
            c="red",
            label="Outside",
        )
        circle = plt.Circle(
            (0, 0), 1, color="green", fill=False, linewidth=2, label="Unit Circle"
        )
        plt.gca().add_artist(circle)
        plt.title(f"{method} 生成的 2 维点的2D可视化")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(markerscale=10)
        plt.grid(True)
        plt.axis("equal")  # 确保比例相等
        plt.show()


def animate_sampling(n, num_samples=1000):
    """为n=2,3创建简单的采样动画"""
    if n not in [2, 3]:
        print("动画仅支持 n=2,3")
        return
    fig = plt.figure(figsize=(8, 6))
    if n == 2:
        ax = fig.add_subplot(111)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        circle = plt.Circle((0, 0), 1, color="green", fill=False, linewidth=2)
        ax.add_artist(circle)
        scat_inside = ax.scatter([], [], c="blue", s=10, label="Inside")
        scat_outside = ax.scatter([], [], c="red", s=10, label="Outside")
        ax.legend(loc="upper right")
    elif n == 3:
        from mpl_toolkits.mplot3d import Axes3D

        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        # 绘制单位球边界
        u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="green", linewidth=0.5, alpha=0.5)
        scat_inside = ax.scatter([], [], [], c="blue", s=1, label="Inside")
        scat_outside = ax.scatter([], [], [], c="red", s=1, label="Outside")
        ax.legend(loc="upper right")

    total_points = 0
    inside_points = 0
    outside_points = 0
    estimated_volume = 0.0

    if n == 2:
        scat_inside.set_offsets(np.empty((0, 2)))
        scat_outside.set_offsets(np.empty((0, 2)))
    elif n == 3:
        scat_inside._offsets3d = ([], [], [])
        scat_outside._offsets3d = ([], [], [])

    def init():
        if n == 2:
            scat_inside.set_offsets(np.empty((0, 2)))
            scat_outside.set_offsets(np.empty((0, 2)))
        elif n == 3:
            scat_inside._offsets3d = ([], [], [])
            scat_outside._offsets3d = ([], [], [])
        return scat_inside, scat_outside

    def update(frame):
        nonlocal total_points, inside_points, outside_points, estimated_volume
        if frame >= num_samples:
            ani.event_source.stop()
            return scat_inside, scat_outside
        if n == 2:
            point = np.random.uniform(-1, 1, 2)
            total_points += 1
            if np.linalg.norm(point) <= 1:
                inside_points += 1
                current_inside = scat_inside.get_offsets()
                if current_inside.size == 0:
                    scat_inside.set_offsets([point])
                else:
                    scat_inside.set_offsets(np.vstack([current_inside, point]))
            else:
                outside_points += 1
                current_outside = scat_outside.get_offsets()
                if current_outside.size == 0:
                    scat_outside.set_offsets([point])
                else:
                    scat_outside.set_offsets(np.vstack([current_outside, point]))
            theoratical_volume = np.pi
            estimated_volume = (inside_points / total_points) * (2**n)
            ax.set_title(
                f"简单均匀采样动画（n={n}）,理论面积:{theoratical_volume:.6f}\n总点数: {total_points}, 内部: {inside_points}, 外部: {outside_points}, 估计面积: {estimated_volume:.4f}"
            )
        elif n == 3:
            point = np.random.uniform(-1, 1, 3)
            total_points += 1
            if np.linalg.norm(point) <= 1:
                inside_points += 1
                current_inside = scat_inside._offsets3d
                scat_inside._offsets3d = (
                    np.append(current_inside[0], point[0]),
                    np.append(current_inside[1], point[1]),
                    np.append(current_inside[2], point[2]),
                )
            else:
                outside_points += 1
                current_outside = scat_outside._offsets3d
                scat_outside._offsets3d = (
                    np.append(current_outside[0], point[0]),
                    np.append(current_outside[1], point[1]),
                    np.append(current_outside[2], point[2]),
                )
            theoratical_volume = np.pi * 4 / 3
            estimated_volume = (inside_points / total_points) * (2**n)
            ax.set_title(
                f"简单均匀采样动画（n={n}）,理论体积:{theoratical_volume:.6f}\n总点数: {total_points}, 内部: {inside_points}, 外部: {outside_points}, 估计体积: {estimated_volume:.4f}"
            )
        return scat_inside, scat_outside

    ani = FuncAnimation(
        fig,
        update,
        frames=num_samples,
        init_func=init,
        blit=False,
        repeat=False,
        interval=1,
    )
    plt.show()


def simple_sampling_analysis(dimensions, num_samples, num_trials=100):
    """简单均匀采样分析，计算估计体积的均值和标准差，并绘制误差棒图"""
    estimates = {d: [] for d in dimensions}
    exact_volumes_dict = {d: exact_volume(d) for d in dimensions}
    print("\n=== 简单均匀采样 ===")
    start_time = time.time()
    for d in tqdm(dimensions, desc="维度"):
        for _ in range(num_trials):
            est = monte_carlo_uniform(d, num_samples)
            estimates[d].append(est)
    end_time = time.time()
    mean_estimates = {d: np.mean(estimates[d]) for d in dimensions}
    std_estimates = {d: np.std(estimates[d]) for d in dimensions}
    print("\n维度\t估计体积均值\t理论体积\t\t相对误差 (%)")
    for d in dimensions:
        exact = exact_volumes_dict[d]
        est_mean = mean_estimates[d]
        relative_error = abs(est_mean - exact) / exact * 100
        print(f"{d}\t{est_mean:.6e}\t{exact:.6e}\t{relative_error:.2f}%")
    plt.figure(figsize=(12, 6))
    dims = dimensions
    means = [mean_estimates[d] for d in dims]
    stds = [std_estimates[d] for d in dims]
    exacts = [exact_volumes_dict[d] for d in dims]
    plt.errorbar(dims, means, yerr=stds, fmt="o", label="估计体积 ± 标准差", capsize=5)
    plt.plot(dims, exacts, "s--", label="理论体积")
    plt.xlabel("维度")
    plt.ylabel("体积")
    plt.title(
        f"简单均匀采样：估计体积 vs 理论体积\n总时间: {end_time - start_time:.2f} 秒"
    )
    plt.legend()
    plt.grid(True)
    plt.xticks(dims)
    plt.show()
    return mean_estimates, std_estimates


def importance_sampling_analysis(dimensions, num_samples, sigma=1.0, num_trials=100):
    """重要性采样分析，计算估计体积的均值和标准差，并绘制误差棒图"""
    exact_volumes_dict = {d: exact_volume(d) for d in dimensions}
    estimates = {d: [] for d in dimensions}
    relative_errors = {d: [] for d in dimensions}
    print("\n=== 重要性采样 ===")
    start_time = time.time()
    total_iterations = len(dimensions) * num_trials
    with tqdm(total=total_iterations, desc="重要性采样进度") as pbar:
        for d in dimensions:
            for _ in range(num_trials):
                est_volume = importance_sampling_volume(d, num_samples, sigma)
                estimates[d].append(est_volume)
                pbar.update(1)
    end_time = time.time()
    mean_estimates = {d: np.mean(estimates[d]) for d in dimensions}
    std_estimates = {d: np.std(estimates[d]) for d in dimensions}
    rel_errors = {
        d: abs(mean_estimates[d] - exact_volumes_dict[d]) / exact_volumes_dict[d] * 100
        for d in dimensions
    }
    print("\n维度\t估计体积均值\t理论体积\t\t相对误差 (%)")
    for d in dimensions:
        exact = exact_volumes_dict[d]
        est_mean = mean_estimates[d]
        rel_error = rel_errors[d]
        print(f"{d}\t{est_mean:.6e}\t{exact:.6e}\t{rel_error:.2f}%")
    plt.figure(figsize=(12, 6))
    dims = dimensions
    means = [mean_estimates[d] for d in dims]
    stds = [std_estimates[d] for d in dims]
    exacts = [exact_volumes_dict[d] for d in dims]
    plt.errorbar(dims, means, yerr=stds, fmt="o", label="估计体积 ± 标准差", capsize=5)
    plt.plot(dims, exacts, "s--", label="理论体积")
    plt.xlabel("维度")
    plt.ylabel("体积")
    plt.title(
        f"重要性采样：估计体积 vs 理论体积\n总时间: {end_time - start_time:.2f} 秒"
    )
    plt.legend()
    plt.grid(True)
    plt.xticks(dims)
    plt.show()
    return mean_estimates, rel_errors


def main():
    while True:
        print("\n=== 高维单位超球体（n-ball）体积估计程序 ===")
        print("请选择操作：")
        print("1. 动画采样（适用于 n=2,3）")
        print("2. 简单均匀采样分析（适用于 n=1~20）")
        print("3. 重要性采样分析（适用于 n=1~20）")
        print("4. 退出")
        choice = input("请输入选择（1/2/3/4）：").strip()

        if choice == "1":
            try:
                n_input = input("请输入维度（2,3）：").strip()
                if not n_input:
                    print("维度不能为空，请输入2或3。")
                    continue
                n = int(n_input)
                if n not in [2, 3]:
                    raise ValueError
                num_samples_input = input(
                    "请输入动画中的采样点数量 [默认：2000]："
                ).strip()
                num_samples = int(num_samples_input) if num_samples_input else 2000
                animate_sampling(n, num_samples)
            except ValueError:
                print("输入无效，请输入2或3。")
                continue

        elif choice == "2":
            try:
                max_dim_input = input("请输入最大维度（1~20） [默认：20]：").strip()
                max_dim = int(max_dim_input) if max_dim_input else 20
                if not (1 <= max_dim <= 20):
                    raise ValueError
                num_samples_input = input(
                    "请输入每次采样的样本数量 [默认：100000]："
                ).strip()
                num_samples = int(num_samples_input) if num_samples_input else 100000
                num_trials_input = input("请输入试验次数 [默认：100]：").strip()
                num_trials = int(num_trials_input) if num_trials_input else 100
                dimensions = list(range(1, max_dim + 1))
            except ValueError:
                print("输入无效，请重新输入。")
                continue
            mean_estimates, std_estimates = simple_sampling_analysis(
                dimensions, num_samples, num_trials
            )
            visualize = (
                input("是否进行采样点可视化？（y/n，默认：n）：").strip().lower()
            )
            if visualize == "y":
                try:
                    d_vis_input = input(
                        f"请输入要可视化的维度（1~{max_dim}） [默认：{max_dim}]："
                    ).strip()
                    d_vis = int(d_vis_input) if d_vis_input else max_dim
                    if not (1 <= d_vis <= max_dim):
                        raise ValueError
                    # 生成部分样本用于可视化
                    samples = np.random.uniform(-1, 1, (1000, d_vis))
                    inside = np.linalg.norm(samples, axis=1) <= 1
                    visualize_sampling(d_vis, samples[inside], "简单均匀采样")
                except ValueError:
                    print("输入无效，跳过可视化。")

        elif choice == "3":
            try:
                max_dim_input = input("请输入最大维度（1~20） [默认：20]：").strip()
                max_dim = int(max_dim_input) if max_dim_input else 20
                if not (1 <= max_dim <= 20):
                    raise ValueError
                num_samples_input = input(
                    "请输入采样的样本数量 [默认：100000]："
                ).strip()
                num_samples = int(num_samples_input) if num_samples_input else 100000
                sigma_input = input("请输入提议分布的标准差 [默认：0.5]：").strip()
                sigma = float(sigma_input) if sigma_input else 0.5
                num_trials_input = input("请输入试验次数 [默认：100]：").strip()
                num_trials = int(num_trials_input) if num_trials_input else 100
                dimensions = list(range(1, max_dim + 1))
            except ValueError:
                print("输入无效，请重新输入。")
                continue
            mean_estimates, rel_errors = importance_sampling_analysis(
                dimensions, num_samples, sigma, num_trials
            )
            visualize = (
                input("是否进行采样点可视化？（y/n，默认：n）：").strip().lower()
            )
            if visualize == "y":
                try:
                    d_vis_input = input(
                        f"请输入要可视化的维度（1~{max_dim}） [默认：{max_dim}]："
                    ).strip()
                    d_vis = int(d_vis_input) if d_vis_input else max_dim
                    if not (1 <= d_vis <= max_dim):
                        raise ValueError
                    # 生成部分样本用于可视化
                    samples = np.random.multivariate_normal(
                        np.zeros(d_vis), (sigma**2) * np.eye(d_vis), 1000
                    )
                    inside = np.sum(samples**2, axis=1) <= 1.0
                    visualize_sampling(d_vis, samples[inside], "重要性采样")
                except ValueError:
                    print("输入无效，跳过可视化。")

        elif choice == "4":
            print("退出程序。")
            sys.exit(0)

        else:
            print("无效的选择，请输入1,2,3或4。")


if __name__ == "__main__":
    main()
