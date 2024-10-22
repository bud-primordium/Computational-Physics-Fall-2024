import numpy as np
import matplotlib.pyplot as plt


# 生成等距节点
def equidistant_nodes(a, b, n):
    return np.linspace(a, b, n)


# 构建差商表
def divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            denominator = x[i] - x[i - j]
            if denominator == 0:
                raise ZeroDivisionError("两个节点的x值相同，无法计算差商。")
            coef[i] = (coef[i] - coef[i - 1]) / denominator
    return coef


# 计算牛顿插值多项式的值（向量化）
def newton_poly_vectorized(coef, x_data, x_vals):
    n = len(coef) - 1
    p = coef[n] * np.ones_like(x_vals)
    for k in range(1, n + 1):
        p = coef[n - k] + (x_vals - x_data[n - k]) * p
    return p


# 病态函数1: f(x) = e^x
def pathological_function_exp(x):
    return np.exp(x)


# 病态函数2: f(x) = sin(100x)
def pathological_function_sin(x):
    return np.sin(100 * x)


# 主函数
def main(function_type="exp"):
    # 根据选择不同的病态函数
    if function_type == "exp":
        f = pathological_function_exp
        a, b = -10, 10  # 区间
        title = "Newton Interpolation: Forward vs. Backward (e^x)"
        ylabel = "$f(x) = e^x$"
        log_scale = True  # 使用对数刻度
    elif function_type == "sin":
        f = pathological_function_sin
        a, b = -0.8, 0.5  # 不对称区间
        title = "Newton Interpolation: Forward vs. Backward (sin(100x))"
        ylabel = "$f(x) = sin(100x)$"
        log_scale = False  # 不使用对数刻度

    # 设置参数
    n = 25  # 节点数量

    # 生成等距节点
    x_nodes = equidistant_nodes(a, b, n)
    y_nodes = f(x_nodes)

    # 前向牛顿插值
    coef_forward = divided_diff(x_nodes, y_nodes)
    x_plot = np.linspace(a, b, 1000)
    y_forward = newton_poly_vectorized(coef_forward, x_nodes, x_plot)

    # 后向牛顿插值（节点顺序反转）
    x_nodes_rev = x_nodes[::-1]
    y_nodes_rev = y_nodes[::-1]
    coef_backward = divided_diff(x_nodes_rev, y_nodes_rev)
    y_backward = newton_poly_vectorized(coef_backward, x_nodes_rev, x_plot)

    # 实际函数值
    y_actual = f(x_plot)

    # 计算相对误差
    epsilon = 1e-12
    relative_error_forward = np.abs(y_forward - y_actual) / (np.abs(y_actual) + epsilon)
    relative_error_backward = np.abs(y_backward - y_actual) / (
        np.abs(y_actual) + epsilon
    )

    # 计算两者相对误差的绝对值差
    relative_diff = relative_error_backward - relative_error_forward

    # 绘图
    plt.figure(figsize=(18, 6))

    # 插值曲线
    plt.subplot(1, 3, 1)
    plt.plot(x_plot, y_actual, label="Actual Function", color="black")
    plt.plot(x_plot, y_forward, label="Forward Newton", linestyle="--")
    plt.plot(x_plot, y_backward, label="Backward Newton", linestyle="-.")
    plt.scatter(x_nodes, y_nodes, color="red", label="Nodes", zorder=5)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale("log")  # 对数刻度用于指数函数

    # 相对误差曲线
    plt.subplot(1, 3, 2)
    plt.plot(
        x_plot,
        relative_error_forward,
        label="Forward Newton Relative Error",
        linestyle="--",
    )
    plt.plot(
        x_plot,
        relative_error_backward,
        label="Backward Newton Relative Error",
        linestyle="-.",
    )
    plt.title("Relative Error: Forward vs. Backward Newton")
    plt.xlabel("x")
    plt.ylabel("Relative Error")
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale("log")  # 对数刻度用于误差展示

    # 两者相对基准的相对误差之差
    plt.subplot(1, 3, 3)
    plt.plot(
        x_plot,
        relative_diff,
        label="Rela.Err Backward - Rela.Err Forward",
        color="purple",
    )
    plt.title("Difference in Relative Errors Between Forward and Backward")
    plt.xlabel("x")
    plt.ylabel("Difference in Relative Errors")
    plt.legend()
    plt.grid(True)
    if log_scale:
        plt.yscale("log")  # 对数刻度用于差异展示

    plt.tight_layout()
    plt.show()


# 测试两个病态函数
if __name__ == "__main__":
    main("exp")  # 使用e^x
    main("sin")  # 使用sin(100x)
