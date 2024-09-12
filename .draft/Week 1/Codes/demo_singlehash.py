import time

def find_diophantine_solution(limit):
    # 记录开始时间
    start_time = time.time()

    # 初始化哈希表，用来存储 a^5 的值
    powers_table = {}

    # 记录构建哈希表开始时间
    build_table_start = time.time()

    # 构建所有 a^5 的表，存储在哈希表中
    for a in range(1, limit):
        powers_table[a] = a**5

    # 记录哈希表构建完成时间
    build_table_end = time.time()

    # 记录查找解的开始时间
    find_solution_start = time.time()

    # 遍历所有可能的 x, y, z, t 组合，查找是否满足条件 x^5 + y^5 + z^5 + t^5 = s
    for x in range(1, limit):
        for y in range(x, limit):
            for z in range(y, limit):
                for t in range(z, limit):
                    sum_of_powers = powers_table[x] + powers_table[y] + powers_table[z] + powers_table[t]

                    # 查找 s 是否为某个数的五次方
                    for s in range(1, limit):
                        if powers_table[s] == sum_of_powers:
                            find_solution_end = time.time()

                            # 输出各个步骤的时间
                            print(f"构建 a^5 表的时间: {build_table_end - build_table_start:.6f} 秒")
                            print(f"查找解的时间: {find_solution_end - find_solution_start:.6f} 秒")
                            print(f"总时间: {find_solution_end - start_time:.6f} 秒")

                            print(f"找到解: {x}^5 + {y}^5 + {z}^5 + {t}^5 = {s}^5")
                            return (x, y, z, t, s)

    print("未找到解")
    return None

# 设定搜索范围
limit = 300
result = find_diophantine_solution(limit)
if result:
    print("解为：", result)
else:
    print("未找到解")
