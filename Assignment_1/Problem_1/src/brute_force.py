"""
@Author: Gilbert Young
@Time: 2024/9/13 21:00
@File_name: brute_force.py
@IDE: Vscode
@Formatter: Black
@Description: This script performs a brute-force search for solutions to the Diophantine equation a^5 + b^5 + c^5 + d^5 = e^5.
"""

import time


def find_diophantine_solution(limit):
    # List of n to n^5; Dictionary of n^5 to n
    pow_5 = [n**5 for n in range(limit + 1)]
    pow5_to_n = {n**5: n for n in range(limit + 1)}

    # Outer four nested loops to find the sum of the fifth powers of four numbers
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            for c in range(b, limit + 1):
                for d in range(c, limit + 1):
                    # Compute the sum of the fifth powers of the first four numbers
                    pow_5_sum = pow_5[a] + pow_5[b] + pow_5[c] + pow_5[d]
                    # Look for pow_5_sum in the fifth power dictionary
                    if pow_5_sum in pow5_to_n:
                        e = pow5_to_n[pow_5_sum]
                        return (a, b, c, d, e)

    return None


start_time = time.time()
# Set the search range
limit = 200
# Call the function to find the solution
result = find_diophantine_solution(limit)
# Measure elapsed time
total_time = time.time() - start_time

# Output the result and time
if result:
    a, b, c, d, e = result
    print(f"Solution: {a}^5 + {b}^5 + {c}^5 + {d}^5 = {e}^5")
else:
    print("No solution found")
print(f"Total time: {total_time:.6f} seconds")
