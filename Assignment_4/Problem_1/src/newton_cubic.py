import numpy as np
import matplotlib.pyplot as plt


# 被插值的函数定义
def f_cos(x):
    return np.cos(x)


def f_rational(x):
    return 1 / (1 + 25 * x**2)


# 生成等距节点
def equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)


# 生成chebyshev节点
def chebyshev_nodes(a, b, n):
    i = np.arange(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * i + 1) * np.pi / (2 * n))


# 生成Leja节点
def leja_nodes(a, b, n, func):
    """基于给定函数在区间 [a, b] 之间生成Leja节点。"""
    x = np.linspace(a, b, 1000)
    nodes = [x[np.argmax(np.abs(func(x)))]]
    for _ in range(1, n):
        distances = np.min(np.abs(x[:, None] - nodes), axis=1)
        next_node = x[np.argmax(distances)]
        nodes.append(next_node)
    return np.array(nodes)


# 构建差商表
def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denominator = x[i] - x[i - j]
            coef[i] = (coef[i] - coef[i - 1]) / denominator
    return coef


# 计算牛顿插值多项式的值（向量化）
def newton_poly_vectorized(coef, x_data, x_vals):
    """在给定点上计算牛顿插值多项式的值。"""
    n = len(coef) - 1
    p = coef[n] * np.ones_like(x_vals)
    for k in range(1, n + 1):
        p = coef[n - k] + (x_vals - x_data[n - k]) * p
    return p


# 解三对角矩阵的Thomas算法
def solve_tridiagonal(A, b):
    """使用Thomas算法解三对角矩阵方程。"""
    n = len(b)
    a_diag = np.zeros(n)
    b_diag = np.zeros(n)
    c_diag = np.zeros(n - 1)

    for i in range(n):
        b_diag[i] = A[i, i]
        if i < n - 1:
            c_diag[i] = A[i, i + 1]
            a_diag[i + 1] = A[i + 1, i]

    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)

    c_prime[0] = c_diag[0] / b_diag[0]
    d_prime[0] = b[0] / b_diag[0]

    for i in range(1, n - 1):
        denom = b_diag[i] - a_diag[i] * c_prime[i - 1]
        c_prime[i] = c_diag[i] / denom
        d_prime[i] = (b[i] - a_diag[i] * d_prime[i - 1]) / denom

    denom = b_diag[-1] - a_diag[-1] * c_prime[-2]
    d_prime[-1] = (b[-1] - a_diag[-1] * d_prime[-2]) / denom

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# 三次样条插值实现（使用Thomas算法求解三对角矩阵）
def cubic_spline_direct(x, y):
    """使用Thomas算法计算三次样条插值的系数，并内部排序节点。"""
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    n = len(x_sorted) - 1  # 区间数量
    h = np.diff(x_sorted)
    a = y_sorted[:-1]

    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    A[0, 0] = 1
    A[-1, -1] = 1
    b_vec[0] = 0
    b_vec[-1] = 0

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b_vec[i] = 3 * (
            (y_sorted[i + 1] - y_sorted[i]) / h[i]
            - (y_sorted[i] - y_sorted[i - 1]) / h[i - 1]
        )

    c = solve_tridiagonal(A, b_vec)

    b_coeff = np.zeros(n)
    d_coeff = np.zeros(n)
    for i in range(n):
        b_coeff[i] = (y_sorted[i + 1] - y_sorted[i]) / h[i] - h[i] * (
            2 * c[i] + c[i + 1]
        ) / 3
        d_coeff[i] = (c[i + 1] - c[i]) / (3 * h[i])

    return a, b_coeff, c[:-1], d_coeff, x_sorted


# 解析三次样条插值
def evaluate_cubic_spline(a, b, c, d, x_data, x_vals):
    """在给定点上解析三次样条插值的值。"""
    y_spline = np.zeros_like(x_vals)
    for i, xv in enumerate(x_vals):
        idx = np.searchsorted(x_data, xv) - 1
        if idx < 0:
            idx = 0
        elif idx >= len(a):
            idx = len(a) - 1
        dx = xv - x_data[idx]
        y_spline[i] = a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3
    return y_spline


# 通用插值实验函数
def run_newton_cubic_comparison(
    func,
    nodes_func,
    n,
    a,
    b,
    title_suffix="",
    node_type="",
    ax_main=None,
    ax_error=None,
):
    """
    运行牛顿插值和三次样条插值，并绘制结果。

    参数:
    - func: 被插值的函数。
    - nodes_func: 生成插值节点的函数。
    - n: 节点数量。
    - a: 区间起点。
    - b: 区间终点。
    - title_suffix: 图表标题的后缀。
    - node_type: 节点类型的名称，用于标签显示。
    - ax_main: 主图的轴对象。
    - ax_error: 误差图的轴对象。
    """
    # 生成节点
    x_data = nodes_func(a, b, n)
    y_nodes = func(x_data)

    # 牛顿插值
    coef = divided_diff(x_data, y_nodes)
    x_vals = np.linspace(a, b, 500)
    y_newton = newton_poly_vectorized(coef, x_data, x_vals)

    # 三次样条插值
    a_spline, b_spline, c_spline, d_spline, x_sorted = cubic_spline_direct(
        x_data, y_nodes
    )
    y_spline = evaluate_cubic_spline(
        a_spline, b_spline, c_spline, d_spline, x_sorted, x_vals
    )

    # 原函数值
    y_true = func(x_vals)

    # 定义绘图元素
    plot_elements = {
        "Original Function": {
            "data": y_true,
            "color": "black",
            "linewidth": 2,
            "linestyle": "-",
        },
        "Newton Interpolation": {
            "data": y_newton,
            "color": "blue",
            "linewidth": 1.5,
            "linestyle": "--",
        },
        "Cubic Spline Interpolation": {
            "data": y_spline,
            "color": "green",
            "linewidth": 1.5,
            "linestyle": "-.",
        },
    }

    error_elements = {
        "Newton Error": {
            "data": np.abs(y_true - y_newton),
            "color": "blue",
            "linestyle": "--",
        },
        "Cubic Spline Error": {
            "data": np.abs(y_true - y_spline),
            "color": "green",
            "linestyle": "-.",
        },
    }

    # 选择是否使用提供的轴对象
    if ax_main is None or ax_error is None:
        fig, (ax_main, ax_error) = plt.subplots(
            1, 2, figsize=(14, 6), constrained_layout=True
        )

    # 绘制主图
    for label, props in plot_elements.items():
        ax_main.plot(
            x_vals,
            props["data"],
            label=label,
            color=props["color"],
            linewidth=props["linewidth"],
            linestyle=props["linestyle"],
        )
    ax_main.scatter(x_data, y_nodes, color="red", label=f"{node_type} Nodes", zorder=5)
    ax_main.set_title(f"{title_suffix} - {node_type} Nodes", fontsize=12)
    ax_main.set_xlabel("x", fontsize=10)
    ax_main.set_ylabel("y", fontsize=10)
    ax_main.legend(fontsize=8)
    ax_main.grid(True)

    # 绘制误差图
    for label, props in error_elements.items():
        ax_error.plot(
            x_vals,
            props["data"],
            label=label,
            color=props["color"],
            linestyle=props["linestyle"],
        )
    ax_error.set_title(f"Error - {node_type} Nodes", fontsize=12)
    ax_error.set_xlabel("x", fontsize=10)
    ax_error.set_ylabel("Error |f(x) - P(x)|", fontsize=10)
    ax_error.legend(fontsize=8)
    ax_error.grid(True)

    # 如果没有提供轴对象，则显示图形
    if ax_main is None or ax_error is None:
        plt.show()

    # 打印最大误差
    print(f"{title_suffix} - {node_type} Nodes")
    print(
        f"Newton Interpolation Max Error: {np.max(error_elements['Newton Error']['data']):.6f}"
    )
    print(
        f"Cubic Spline Interpolation Max Error: {np.max(error_elements['Cubic Spline Error']['data']):.6f}\n"
    )


# 主程序执行实验
if __name__ == "__main__":
    n = 10  # 默认节点数量

    # 实验1：cos(x) 使用等距节点
    print("### Experiment 1: cos(x) with Equidistant Nodes ###\n")
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    run_newton_cubic_comparison(
        func=f_cos,
        nodes_func=equidistant_nodes,
        n=n,
        a=0,
        b=np.pi,
        title_suffix="cos(x)",
        node_type="Equidistant",
        ax_main=axes1[0],
        ax_error=axes1[1],
    )

    # 实验2：比较不同节点类型（等距、chebyshev、Leja）在插值 1/(1 + 25x^2) 时的表现
    print("### Experiment 2: 1/(1 + 25x²) with Different Node Types ###\n")
    node_types = {
        "Equidistant": equidistant_nodes,
        "Chebyshev": chebyshev_nodes,
        "Leja": lambda a, b, n: leja_nodes(a, b, n, f_rational),
    }

    fig2, axes2 = plt.subplots(
        len(node_types), 2, figsize=(16, 18), constrained_layout=True
    )

    for idx, (name, nodes_func) in enumerate(node_types.items()):
        # 对Leja节点使用n=11，其他节点使用n=10
        current_n = 11 if name == "Leja" else n
        print(f"### Node Type: {name} ###\n")
        run_newton_cubic_comparison(
            func=f_rational,
            nodes_func=nodes_func,
            n=current_n,
            a=-1,
            b=1,
            title_suffix="1/(1 + 25x²)",
            node_type=name,
            ax_main=axes2[idx, 0],
            ax_error=axes2[idx, 1],
        )

    plt.show()
