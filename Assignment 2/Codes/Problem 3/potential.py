"""
@Author: Gilbert Young
@Time: 2024/09/20 20:41
@File_name: potential.py
@IDE: Vscode
@Formatter: Black
@Description: This script visualizes the bisection method for finding roots of the function f(x) = x^3 - 5x + 3. 
               It dynamically updates the plot to show the current interval and root approximation during the iterations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Define the constant z0
z0 = 3.240175521


# Define the functions
def f1(z):
    return np.tan(z)


def f2(z):
    return -1 / np.tan(z)


def f3(z):
    return np.sqrt((z0 / z) ** 2 - 1)


# Function for finding the intersections
def intersection1(z):
    return f1(z) - f3(z)


def intersection2(z):
    return f2(z) - f3(z)


# Generate z values for plotting, avoid dividing by zero or tangents of multiples of pi/2
z_values = np.linspace(0.1, z0, 400)
z_values = z_values[(z_values % (np.pi / 2)) != 0]

# Solve for intersections, exclude near multiples of pi/2 and provide good starting points
intersections_f1_f3 = fsolve(intersection1, [1.0, 3.2])
intersections_f2_f3 = fsolve(intersection2, [2.0])

# Filter out bad solutions close to undefined points
good_intersections_f1_f3 = [
    i for i in intersections_f1_f3 if not np.isclose(i % (np.pi / 2), 0)
]
good_intersections_f2_f3 = [
    i for i in intersections_f2_f3 if not np.isclose(i % (np.pi / 2), 0)
]

# Plot the functions
plt.plot(z_values, f1(z_values), label="tan(z)", linewidth=2)
plt.plot(z_values, f2(z_values), label="-cot(z)", linewidth=2)
plt.plot(z_values, f3(z_values), label="sqrt((z0/z)^2 - 1)", linewidth=2)

# Mark intersections on the plot
for z in good_intersections_f1_f3:
    plt.plot(z, f1(z), "ro")  # Red circle
    plt.annotate(
        f"({z:.2f}, {f1(z):.2f})",
        (z, f1(z)),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

for z in good_intersections_f2_f3:
    plt.plot(z, f2(z), "go")  # Green circle
    plt.annotate(
        f"({z:.2f}, {f2(z):.2f})",
        (z, f2(z)),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

# Add horizontal and vertical lines at zero
plt.axhline(0, color="black", lw=0.5, ls="--")
plt.axvline(0, color="black", lw=0.5, ls="--")

# Set the limits for the x and y axes
plt.xlim(0, 4)
plt.ylim(-10, 10)

# Label the axes
plt.xlabel("z")
plt.ylabel("f(z)")

# Add a title to the plot
plt.title("Functions Plot")

# Add a legend to the plot
plt.legend()

# Add a grid to the plot
plt.grid()

# Display the plot
plt.show()

# Output the intersection points
good_intersections_f1_f3, good_intersections_f2_f3
