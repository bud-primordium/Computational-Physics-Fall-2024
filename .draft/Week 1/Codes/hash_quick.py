import time

def find_diophantine_solution(limit):
    # 初始化两个哈希表
    ab_sum_table = {}
    e_power_table = {}

    # 记录开始时间
    start_time = time.time()

    # 构建 a^5 + b^5 的哈希表
    build_ab_table_start = time.time()
    for a in range(1, limit):
        for b in range(a, limit):
            power_sum = a**5 + b**5
            if power_sum not in ab_sum_table:
                ab_sum_table[power_sum] = [(a, b)]
            else:
                ab_sum_table[power_sum].append((a, b))
    build_ab_table_end = time.time()

    # 构建 e^5 的哈希表
    build_e_table_start = time.time()
    for e in range(1, limit):
        e_power_table[e] = e**5
    build_e_table_end = time.time()

    # 查找 c^5 + d^5 的组合是否能匹配 a^5 + b^5
    find_solution_start = time.time()
    for c in range(1, limit):
        for d in range(c, limit):
            cd_sum = c**5 + d**5
            for e, e_power in e_power_table.items():
                if (e_power - cd_sum) in ab_sum_table:
                    for (a, b) in ab_sum_table[e_power - cd_sum]:
                        find_solution_end = time.time()
                        total_time = find_solution_end - start_time

                        # 输出结果和时间
                        print(f"构建 a^5 + b^5 表的时间: {build_ab_table_end - build_ab_table_start:.6f} 秒")
                        print(f"构建 e^5 表的时间: {build_e_table_end - build_e_table_start:.6f} 秒")
                        print(f"查找解的时间: {find_solution_end - find_solution_start:.6f} 秒")
                        print(f"总时间: {total_time:.6f} 秒")
                        print(f"找到解: {a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
                        return (a, b, c, d, e)

    print("未找到解")
    return None

# 设定搜索范围
limit = 250
result = find_diophantine_solution(limit)
if result:
    print("解为：", result)
else:
    print("未找到解")
