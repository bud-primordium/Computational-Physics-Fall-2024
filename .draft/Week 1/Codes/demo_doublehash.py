import time

def find_first_diophantine_solution(limit):
    # 记录开始时间
    start_time = time.time()

    # 初始化哈希表，用来存储 a^5 + b^5 的结果
    powers_hash_table = {}

    # 记录构建哈希表开始时间
    build_table_start = time.time()

    # 计算 a^5 + b^5 并存入哈希表
    for a in range(1, limit):
        for b in range(a, limit):  # b >= a 避免重复
            power_sum = a**5 + b**5
            if power_sum not in powers_hash_table:
                powers_hash_table[power_sum] = [(a, b)]
            else:
                powers_hash_table[power_sum].append((a, b))

    # 记录哈希表构建完成时间
    build_table_end = time.time()

    # 遍历所有可能的 c 和 d，计算 e^5 - (c^5 + d^5)，并查找是否匹配 a^5 + b^5
    find_solution_start = time.time()
    
    for e in range(1, limit):
        e_power = e**5
        for c in range(1, limit):
            for d in range(c, limit):  # d >= c 避免重复
                cd_power_sum = c**5 + d**5
                target_sum = e_power - cd_power_sum
                if target_sum in powers_hash_table:
                    # 找到第一个解
                    for (a, b) in powers_hash_table[target_sum]:
                        find_solution_end = time.time()
                        total_time = find_solution_end - start_time

                        # 输出各个步骤的时间
                        print(f"构建哈希表时间: {build_table_end - build_table_start:.6f} 秒")
                        print(f"查找解的时间: {find_solution_end - find_solution_start:.6f} 秒")
                        print(f"总时间: {total_time:.6f} 秒")

                        print(f"找到解: {a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
                        return (a, b, c, d, e)

    # 如果没有找到解
    print("未找到解")
    return None

# 设定搜索范围
limit = 300
result = find_first_diophantine_solution(limit)
if result:
    print("解为：", result)
else:
    print("未找到解")
