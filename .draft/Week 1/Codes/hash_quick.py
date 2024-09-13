import time
from concurrent.futures import ThreadPoolExecutor

def find_diophantine_solution(limit):
    # 预计算每个整数的五次幂并存储与其相关的索引
    power_5 = {i**5: i for i in range(1, limit)}

    # 使用字典存储两个数的五次幂之和与其索引的映射
    sum2 = {}

    # 定义函数以并行计算五次幂之和
    def compute_sum(i):
        results = {}
        a5 = i**5
        for j in range(i, limit):
            results[a5 + j**5] = (i, j)
        return results

    # 使用线程池并行构建 sum2 字典
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_sum, i) for i in range(1, limit)]
        for future in futures:
            sum2.update(future.result())

    # 对 sum2 的键进行排序，以便后续快速查找
    sk = sorted(sum2.keys())
    
    # 遍历所有的五次幂，查找是否存在符合条件的五个数字
    for p in sorted(power_5.keys()):
        for s in sk:
            if p <= s:
                break  # 终止循环，优化性能
            if p - s in sum2:
                return (power_5[p], sum2[s], sum2[p-s])
    return None

start_time = time.time()
# 设定搜索范围
limit = 200
# 调用函数查找解
result = find_diophantine_solution(limit)
# 计时
total_time = time.time() - start_time

# 输出结果和时间
if result:
    e, (a, b), (c, d)= result
    print(f"解为：{a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
else:
    print("未找到解")
print(f"总时间: {total_time:.6f} 秒")
