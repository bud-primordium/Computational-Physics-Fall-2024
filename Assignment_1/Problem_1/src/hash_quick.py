"""
@Author: Gilbert Young
@Time: 2024/9/13 21:00
@File_name: hash_quick.py
@IDE: Vscode
@Formatter: Black
@Description: This script performs a parallel brute-force search for solutions to the Diophantine equation a^5 + b^5 + c^5 + d^5 = e^5. 
               It uses hashing to efficiently find solutions and leverages multi-threading to precompute fifth powers.
"""

import time
from concurrent.futures import ThreadPoolExecutor


def find_diophantine_solution(limit):
    # Precompute the fifth power of each integer and store it with its corresponding index
    power_5 = {i**5: i for i in range(1, limit)}

    # Use a dictionary to store sums of fifth powers and their corresponding indices
    sum2 = {}

    # Define a function to compute sums of fifth powers in parallel
    def compute_sum(i):
        results = {}
        a5 = i**5
        for j in range(i, limit):
            results[a5 + j**5] = (i, j)
        return results

    # Use a thread pool to parallelize the computation of sum2 dictionary
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_sum, i) for i in range(1, limit)]
        for future in futures:
            sum2.update(future.result())

    # Sort the keys of sum2 for faster lookup
    sk = sorted(sum2.keys())

    # Iterate over all fifth powers to find if there exist five numbers that satisfy the equation
    for p in sorted(power_5.keys()):
        for s in sk:
            if p <= s:
                break  # Terminate loop to optimize performance
            if p - s in sum2:
                return (power_5[p], sum2[s], sum2[p - s])
    return None


start_time = time.time()
# Define the search limit
limit = 200
# Call the function to find the solution
result = find_diophantine_solution(limit)
# Measure the elapsed time
total_time = time.time() - start_time

# Output the result and elapsed time
if result:
    e, (a, b), (c, d) = result
    print(f"Solution: {a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
else:
    print("No solution found")
print(f"Total time: {total_time:.6f} seconds")
