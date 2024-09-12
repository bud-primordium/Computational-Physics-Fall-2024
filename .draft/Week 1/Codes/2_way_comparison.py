import time
from itertools import combinations_with_replacement

# Function to build triple power sum table and perform two loops to search (scheme 1)
def scheme1_multiple_solutions(N):
    # Step 1: Build triple power sum table a^5 + b^5 + c^5
    triple_power_sum = {}
    for a, b, c in combinations_with_replacement(range(1, N), 3):
        sum_abc = a**5 + b**5 + c**5
        if sum_abc not in triple_power_sum:
            triple_power_sum[sum_abc] = [(a, b, c)]
        else:
            triple_power_sum[sum_abc].append((a, b, c))

    solutions = []

    # Step 2: Search using two loops (for e and d)
    for e in range(1, N):
        e5 = e**5
        for d in range(1, e):
            d5 = d**5
            difference = e5 - d5
            if difference in triple_power_sum:
                for abc_triple in triple_power_sum[difference]:
                    solutions.append((abc_triple, (d, e)))

    return solutions

# Function to build double power sum table and perform three loops to search (scheme 2)
def scheme2_multiple_solutions(N):
    # Step 1: Build double power sum table a^5 + b^5
    double_power_sum = {}
    for a, b in combinations_with_replacement(range(1, N), 2):
        sum_ab = a**5 + b**5
        if sum_ab not in double_power_sum:
            double_power_sum[sum_ab] = [(a, b)]
        else:
            double_power_sum[sum_ab].append((a, b))

    solutions = []

    # Step 2: Search using three loops (for e, d, c)
    for e in range(1, N):
        e5 = e**5
        for d in range(1, e):
            d5 = d**5
            for c in range(1, d):
                difference = e5 - d5 - c**5
                if difference in double_power_sum:
                    for ab_pair in double_power_sum[difference]:
                        solutions.append((ab_pair, c, (d, e)))

    return solutions

# Function to measure execution time of each scheme
def measure_time(scheme, N):
    start_time = time.time()
    result = scheme(N)
    end_time = time.time()
    return result, end_time - start_time

# Test with N = 250 for both schemes
N = 250
result1, time1 = measure_time(scheme1_multiple_solutions, N)
result2, time2 = measure_time(scheme2_multiple_solutions, N)

print(f"Scheme 1 found {len(result1)} solutions in {time1:.2f} seconds.")
for solution in result1[:5]:  # Print only the first 5 solutions for brevity
    print(solution)

print(f"\nScheme 2 found {len(result2)} solutions in {time2:.2f} seconds.")
for solution in result2[:5]:  # Print only the first 5 solutions for brevity
    print(solution)
