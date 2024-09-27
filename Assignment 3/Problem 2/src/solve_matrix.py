import numpy as np

matrix = np.loadtxt("pi_81.in")

A = matrix[:, :-1]  # augmented_matri (81x81)
b = matrix[:, -1]  # const vector (81x1)

# Solve
solution = np.linalg.solve(A, b)

# Output
np.set_printoptions(precision=4)
print("Solution to the system (rounded to 4 decimal places):")
print(solution)

# Save to file
np.savetxt("solution.out", solution, fmt="%.4f")
