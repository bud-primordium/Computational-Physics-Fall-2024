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
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
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
    """可视化采样点分布"""
    if n > 3:
        # 高维数据使用PCA降维到2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(points)
        plt.figure(figsize=(10, 8))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5, s=1)
        plt.title(
            f"{method}生成的{n}维点的2D投影\n(解释方差比: {pca.explained_variance_ratio_.sum():.2%})"
        )
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.grid(True)
        plt.axis("equal")

        # 添加距离分布直方图
        plt.figure(figsize=(8, 6))
        distances = np.linalg.norm(points, axis=1)
        plt.hist(distances, bins=50, density=True)
        plt.title(f"{n}维空间中点到原点的距离分布")
        plt.xlabel("距离")
        plt.ylabel("密度")
        plt.grid(True)
        plt.show()

    elif n == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 10))
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
        ax.set_title(f"{method}生成的3维点分布")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("x3")
        ax.legend()
        plt.show()

    elif n == 2:
        plt.figure(figsize=(10, 10))
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
        plt.title(f"{method}生成的2维点分布")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.legend(markerscale=10)
        plt.grid(True)
        plt.axis("equal")
        plt.show()


def animate_sampling(n, num_samples=2000):
    """动态展示采样过程"""
    if n not in [2, 3]:
        print("动画仅支持2维和3维可视化")
        return

    fig = plt.figure(figsize=(10, 8))
    if n == 2:
        ax = fig.add_subplot(111)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        circle = plt.Circle((0, 0), 1, color="green", fill=False, linewidth=2)
        ax.add_artist(circle)
        scat_inside = ax.scatter([], [], c="blue", s=10, label="Inside")
        scat_outside = ax.scatter([], [], c="red", s=10, label="Outside")
        ax.legend(loc="upper right")
    else:  # n == 3
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
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
    theoretical_volume = np.pi if n == 2 else 4 * np.pi / 3

    def init():
        if n == 2:
            scat_inside.set_offsets(np.empty((0, 2)))
            scat_outside.set_offsets(np.empty((0, 2)))
        else:
            scat_inside._offsets3d = ([], [], [])
            scat_outside._offsets3d = ([], [], [])
        return scat_inside, scat_outside

    def update(frame):
        nonlocal total_points, inside_points
        if frame >= num_samples:
            ani.event_source.stop()
            return scat_inside, scat_outside

        if n == 2:
            point = np.random.uniform(-1, 1, 2)
            total_points += 1
            if np.linalg.norm(point) <= 1:
                inside_points += 1
                current_inside = scat_inside.get_offsets()
                new_points = (
                    np.vstack([current_inside, point])
                    if current_inside.size > 0
                    else np.array([point])
                )
                scat_inside.set_offsets(new_points)
            else:
                current_outside = scat_outside.get_offsets()
                new_points = (
                    np.vstack([current_outside, point])
                    if current_outside.size > 0
                    else np.array([point])
                )
                scat_outside.set_offsets(new_points)
        else:  # n == 3
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
                current_outside = scat_outside._offsets3d
                scat_outside._offsets3d = (
                    np.append(current_outside[0], point[0]),
                    np.append(current_outside[1], point[1]),
                    np.append(current_outside[2], point[2]),
                )

        estimated_volume = (inside_points / total_points) * (2**n)
        ax.set_title(
            f"Monte Carlo采样过程 (n={n})\n"
            f"理论体积: {theoretical_volume:.6f}\n"
            f"估计体积: {estimated_volume:.6f}\n"
            f"采样点数: {total_points}, 内部点数: {inside_points}"
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


def analyze_volumes(
    max_dim=20, num_samples=100000, num_trials=100, method="uniform", sigma=0.5
):
    """分析不同维度的体积和误差"""
    dimensions = range(1, max_dim + 1)
    results = {
        "dims": list(dimensions),
        "exact": [],
        "estimated": [],
        "std_dev": [],
        "rel_error": [],
        "computation_time": [],
    }

    print("\n分析中...")
    for d in tqdm(dimensions):
        # 计算理论体积
        exact = exact_volume(d)
        results["exact"].append(exact)

        # 多次采样估计
        start_time = time.time()
        if method == "uniform":
            estimates = [monte_carlo_uniform(d, num_samples) for _ in range(num_trials)]
        else:  # importance sampling
            estimates = [
                importance_sampling_volume(d, num_samples, sigma)
                for _ in range(num_trials)
            ]
        comp_time = time.time() - start_time

        mean_est = np.mean(estimates)
        std_dev = np.std(estimates)
        rel_error = abs(mean_est - exact) / exact * 100

        results["estimated"].append(mean_est)
        results["std_dev"].append(std_dev)
        results["rel_error"].append(rel_error)
        results["computation_time"].append(comp_time)

    return results


def plot_analysis(results, method):
    """绘制分析结果"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"n维单位超球体体积分析 ({method})", fontsize=16)

    dims = results["dims"]

    # 所有子图使用整数刻度
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 1. 体积对比图
    ax1.plot(dims, results["exact"], "b-", label="理论体积")
    ax1.errorbar(
        dims,
        results["estimated"],
        yerr=results["std_dev"],
        fmt="ro",
        label="估计体积±标准差",
        capsize=3,
    )
    ax1.set_title("体积比较")
    ax1.set_xlabel("维度")
    ax1.set_ylabel("体积")
    ax1.legend()
    ax1.grid(True)

    # 2. 相对误差图
    ax2.plot(dims, results["rel_error"], "r-o")
    ax2.set_title("相对误差随维度变化")
    ax2.set_xlabel("维度")
    ax2.set_ylabel("相对误差 (%)")
    ax2.grid(True)

    # 3. 计算时间图
    ax3.plot(dims, results["computation_time"], "g-o")
    ax3.set_title("计算时间随维度变化")
    ax3.set_xlabel("维度")
    ax3.set_ylabel("时间 (秒)")
    ax3.grid(True)

    # 4. 相邻维度体积比率分析
    volume_ratio = [
        v2 / v1 for v1, v2 in zip(results["exact"][:-1], results["exact"][1:])
    ]
    ax4.plot(dims[1:], volume_ratio, "b-o")
    ax4.set_title("相邻维度体积比率 (Vn+1/Vn)")
    ax4.set_xlabel("维度 n")
    ax4.set_ylabel("Vn+1/Vn")
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

    # 打印详细结果
    print("\n详细结果:")
    print("维度\t理论体积\t估计体积\t相对误差(%)\t计算时间(s)")
    print("-" * 70)
    for i, d in enumerate(dims):
        print(
            f"{d}\t{results['exact'][i]:.6e}\t{results['estimated'][i]:.6e}\t"
            f"{results['rel_error'][i]:.2f}\t{results['computation_time'][i]:.3f}"
        )


def main():
    while True:
        print("\n=== 高维单位超球体（n-ball）体积估计程序 ===")
        print("请选择操作：")
        print("1. 动画演示采样过程（适用于n=2,3）")
        print("2. 简单均匀采样分析（适用于n=1~20）")
        print("3. 重要性采样分析（适用于n=1~20）")
        print("4. 退出")

        choice = input("\n请输入选择（1/2/3/4）：").strip()

        if choice == "1":
            try:
                n = int(input("请输入维度（2或3）：").strip())
                if n not in [2, 3]:
                    print("维度必须是2或3")
                    continue
                num_samples = int(
                    input("请输入采样点数量 [默认1919]：").strip() or "1919"
                )
                animate_sampling(n, num_samples)
            except ValueError:
                print("输入无效，请重试。")

        elif choice == "2":
            try:
                max_dim = int(
                    input("请输入最大维度（1~20） [默认20]：").strip() or "20"
                )
                if not (1 <= max_dim <= 20):
                    print("最大维度应在1到20之间")
                    continue
                num_samples = int(
                    input("请输入每次采样的样本数量 [默认114514]：").strip() or "114514"
                )
                num_trials = int(input("请输入试验次数 [默认100]：").strip() or "100")

                results = analyze_volumes(
                    max_dim, num_samples, num_trials, method="uniform"
                )
                plot_analysis(results, "简单均匀采样")

                visualize = (
                    input("是否进行采样点可视化？（y/n，默认：n）：").strip().lower()
                )
                if visualize == "y":
                    try:
                        d_vis = int(
                            input(
                                f"请输入要可视化的维度（1~{max_dim}） [默认：5]："
                            ).strip()
                            or "5"
                        )
                        if not (1 <= d_vis <= max_dim):
                            raise ValueError
                        points = np.random.uniform(-1, 1, (10000, d_vis))
                        inside = np.sum(points**2, axis=1) <= 1
                        visualize_sampling(d_vis, points[inside], "简单均匀采样")
                    except ValueError:
                        print("输入无效，跳过可视化。")
            except ValueError:
                print("输入无效，请重试。")

        elif choice == "3":
            try:
                max_dim = int(
                    input("请输入最大维度（1~20） [默认20]：").strip() or "20"
                )
                if not (1 <= max_dim <= 20):
                    print("最大维度应在1到20之间")
                    continue
                num_samples = int(
                    input("请输入采样的样本数量 [默认114514]：").strip() or "114514"
                )
                sigma = float(
                    input("请输入提议分布的标准差 [默认0.5]：").strip() or "0.5"
                )
                num_trials = int(input("请输入试验次数 [默认100]：").strip() or "100")

                results = analyze_volumes(
                    max_dim, num_samples, num_trials, method="importance", sigma=sigma
                )
                plot_analysis(results, "重要性采样")

                visualize = (
                    input("是否进行采样点可视化？（y/n，默认：n）：").strip().lower()
                )
                if visualize == "y":
                    try:
                        d_vis = int(
                            input(
                                f"请输入要可视化的维度（1~{max_dim}） [默认：5]："
                            ).strip()
                            or "5"
                        )
                        if not (1 <= d_vis <= max_dim):
                            raise ValueError
                        samples = np.random.multivariate_normal(
                            np.zeros(d_vis), (sigma**2) * np.eye(d_vis), 10000
                        )
                        inside = np.sum(samples**2, axis=1) <= 1.0
                        visualize_sampling(d_vis, samples[inside], "重要性采样")
                    except ValueError:
                        print("输入无效，跳过可视化。")
            except ValueError:
                print("输入无效，请重试。")

        elif choice == "4" or choice.lower() == "q":
            print("程序已退出。")
            break

        else:
            print("无效的选择，请输入1、2、3或4。")


if __name__ == "__main__":
    main()
