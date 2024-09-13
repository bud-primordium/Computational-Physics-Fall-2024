import time

def find_diophantine_solution(limit):
    # 列表为n to n^5; 字典为n^5 to n
    pow_5 = [n**5 for n in range(limit + 1)]
    pow5_to_n = {n**5: n for n in range(limit + 1)}

    # 外层四重循环来寻找四个数字的五次幂之和
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            for c in range(b, limit + 1):
                for d in range(c, limit + 1):
                    # 计算前四个数的五次幂之和
                    pow_5_sum = pow_5[a] + pow_5[b] + pow_5[c] + pow_5[d]
                    # 在五次幂字典中寻找pow_5_sum
                    if pow_5_sum in pow5_to_n:
                        e = pow5_to_n[pow_5_sum]
                        return (a, b, c, d, e)
    
    return None

start_time = time.time()
# 设定搜索范围
limit = 200
# 调用函数查找解
result = find_diophantine_solution(limit)
# 计时
total_time = time.time() - start_time

# 输出结果和用时
if result:
    a, b, c, d, e = result
    print(f"解为：{a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
else:
    print("未找到解")
print(f"总时间: {total_time:.6f} 秒")
