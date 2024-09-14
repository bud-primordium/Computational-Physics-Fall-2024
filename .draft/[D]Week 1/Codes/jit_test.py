import time
import numpy as np
from numba import jit

# 不使用 JIT 的普通 Python 函数
def normal_sum_of_squares(n):
    total = 0
    for i in range(n):
        total += i ** 2
    return total

# 使用 Numba 进行 JIT 编译
@jit(nopython=True)
def jit_sum_of_squares(n):
    total = 0
    for i in range(n):
        total += i ** 2
    return total


# 测试 JIT 编译的 Python 函数
n = 10000000
start_time = time.time()
print(jit_sum_of_squares(n))
print("JIT compiled execution time: %s seconds" % (time.time() - start_time))


# 测试普通 Python 函数

start_time = time.time()
print(normal_sum_of_squares(n))
print("Normal Python execution time: %s seconds" % (time.time() - start_time))