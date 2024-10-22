import numpy as np
import matplotlib.pyplot as plt

# 步骤 1：定义数据
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
T = [14.6, 18.5, 36.6, 30.8, 59.2, 60.1, 62.2, 79.4, 99.9]

# 转换为NumPy数组
x = np.array(x)
T = np.array(T)

# 步骤 2：构建设计矩阵
# 线性拟合设计矩阵
A_linear = np.vstack([np.ones(len(x)), x]).T

# 二次拟合设计矩阵
A_quadratic = np.vstack([np.ones(len(x)), x, x**2]).T

# 步骤 3：执行SVD分解
# 线性设计矩阵的SVD
U_linear, S_linear, VT_linear = np.linalg.svd(A_linear, full_matrices=False)

# 二次设计矩阵的SVD
U_quadratic, S_quadratic, VT_quadratic = np.linalg.svd(A_quadratic, full_matrices=False)


# 步骤 4：计算伪逆矩阵
def compute_pseudo_inverse(U, S, VT):
    """
    计算矩阵的伪逆 A^+ = V Σ^+ U^T
    """
    # 取奇异值的倒数，处理接近零的奇异值以避免除零
    S_inv = np.array([1 / s if s > 1e-10 else 0 for s in S])

    # 构建Σ^+矩阵
    Sigma_inv = np.diag(S_inv)

    # 计算伪逆 A^+ = V Σ^+ U^T
    A_pseudo_inv = VT.T @ Sigma_inv @ U.T
    return A_pseudo_inv


# 计算线性设计矩阵的伪逆
A_linear_pseudo_inv = compute_pseudo_inverse(U_linear, S_linear, VT_linear)

# 计算二次设计矩阵的伪逆
A_quadratic_pseudo_inv = compute_pseudo_inverse(U_quadratic, S_quadratic, VT_quadratic)

# 步骤 5：求解最小二乘解
# 线性拟合参数 [a, b]
beta_linear = A_linear_pseudo_inv @ T
a_linear, b_linear = beta_linear
print(f"Linear fit result: T(x) = {a_linear:.4f} + {b_linear:.4f}x")

# 二次拟合参数 [a, b, c]
beta_quadratic = A_quadratic_pseudo_inv @ T
a_quadratic, b_quadratic, c_quadratic = beta_quadratic
print(
    f"Quadratic fit result: T(x) = {a_quadratic:.4f} + {b_quadratic:.4f}x + {c_quadratic:.4f}x²"
)

# 步骤 6：计算拟合值和残差
# 线性拟合值
T_fit_linear = a_linear + b_linear * x

# 二次拟合值
T_fit_quadratic = a_quadratic + b_quadratic * x + c_quadratic * x**2

# 计算残差平方和
RSS_linear = np.sum((T - T_fit_linear) ** 2)
RSS_quadratic = np.sum((T - T_fit_quadratic) ** 2)

print(f"Residual Sum of Squares (RSS) for linear fit: {RSS_linear:.4f}")
print(f"Residual Sum of Squares (RSS) for quadratic fit: {RSS_quadratic:.4f}")

# 计算决定系数 R²
T_mean = np.mean(T)
SST = np.sum((T - T_mean) ** 2)

R2_linear = 1 - RSS_linear / SST
R2_quadratic = 1 - RSS_quadratic / SST

print(f"R² for linear fit: {R2_linear:.4f}")
print(f"R² for quadratic fit: {R2_quadratic:.4f}")

# 步骤 7：可视化结果
plt.scatter(x, T, label="Data Points", color="blue")

plt.plot(x, T_fit_linear, label="Linear Fit", color="red")

plt.plot(x, T_fit_quadratic, label="Quadratic Fit", color="green")

plt.xlabel("Position x (cm)")
plt.ylabel("Temperature T (°C)")
plt.title("Least Squares Fit Results (Using SVD)")
plt.legend()
plt.grid(True)
plt.show()
