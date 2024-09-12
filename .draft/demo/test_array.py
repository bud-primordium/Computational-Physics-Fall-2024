import numpy as np

dims = (2, 3, 4)
data = np.array([1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12,
                 13, 17, 21, 14, 18, 22, 15, 19, 23, 16, 20, 24])
A = data.reshape(dims, order='F')

# 访问与Fortran的 A(1,2,3) 对应的NumPy索引
element = A[0, 1, 2]  # 注意NumPy索引从0开始
print(element)  # 输出: 15
