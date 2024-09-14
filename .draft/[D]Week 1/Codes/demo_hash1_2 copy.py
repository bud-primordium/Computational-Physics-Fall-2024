import time

def construct_powers_table(limit):
    # 初始化哈希表，用来存储 a^5 + b^5 的结果
    ab_sum_table = {}

    # 构建 a^5 + b^5 的哈希表
    for a in range(1, limit):
        for b in range(a, limit):
            power_sum = a**5 + b**5
            if power_sum not in ab_sum_table:
                ab_sum_table[power_sum] = [(a, b)]
            else:
                ab_sum_table[power_sum].append((a, b))

    return ab_sum_table

def construct_power_table(limit):
    # 初始化哈希表，用来存储 a^5 的结果
    power_table = {}

    # 构建 a^5 的哈希表
    for a in range(1, limit):
        power_table[a] = a**5

    return power_table

def optimized_find_diophantine_solution(limit):
    # 构建双幂和表和单幂表
    ab_sum_table = construct_powers_table(limit)
    power_table = construct_power_table(limit)

    # 遍历 e^5，查找解
    for e in range(1, limit):
        e_power = power_table[e]  # e^5 的值

        # 限制 sum1 <= e^5 / 2
        for sum1 in ab_sum_table:
            if sum1 > e_power / 2:
                continue  # 如果 sum1 大于 e^5 / 2，就跳过

            # 计算差值，看看它是否在表中
            target_sum = e_power - sum1
            if target_sum in ab_sum_table:
                # 找到匹配项，输出对应的 (a, b) 和 (c, d)
                for (a, b) in ab_sum_table[sum1]:
                    for (c, d) in ab_sum_table[target_sum]:
                        print(f"找到解: {a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
                        return (a, b, c, d, e)

    print("未找到解")
    return None

# 测量运行时间
start_time = time.time()
limit = 200  # 设置一个合理的 limit
result = optimized_find_diophantine_solution(limit)
end_time = time.time()
print(f"优化方案运行时间: {end_time - start_time:.6f} 秒")
