"""
@Author: Gilbert Young
@Time: 2024/09/19 01:47
@File_name: bisection_method_visualization.py
@IDE: Vscode
@Formatter: Black
@Description: This script visualizes the bisection method for finding roots of the function f(x) = x^3 - 5x + 3. 
               It dynamically updates the plot to show the current interval and root approximation during the iterations.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import sys


# Define the function f(x) = x^3 - 5x + 3
def f(x):
    return x**3 - 5 * x + 3


# Implement the bisection method, returning a list of all iteration results
def bisection_method(a, b, tol, max_iter):
    iterations = []
    fa = f(a)
    fb = f(b)
    if fa * fb >= 0:
        raise ValueError(
            "f(a) and f(b) must have opposite signs to ensure a root exists within the interval."
        )

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        iterations.append((a, b, c, fa, fb, fc))
        print(f"Iteration {i}: a = {a:.5f}, b = {b:.5f}, c = {c:.5f}, f(c) = {fc:.5f}")
        if (b - a) / 2 < tol:
            break
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return iterations


# Initialize the plot
def init_plot(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title("Bisection Method Visualization")
    ax.grid(True)
    x = np.linspace(x_min, x_max, 400)
    y = f(x)
    ax.plot(x, y, label=r"$f(x) = x^3 - 5x + 3$", color="black", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()


# Update the plot
def update_plot(frame, gray_lines, current_points, ax, text, min_margin=0.1):
    a, b, c, fa, fb, fc = frame
    a_gray, b_gray, c_gray = gray_lines

    # Update gray trajectories for previous points
    for line, x in zip((a_gray, b_gray, c_gray), (a, b, c)):
        line_x, line_y = line.get_data()
        line.set_data(np.append(line_x, x), np.append(line_y, f(x)))

    # Update current points
    for point, x in zip(current_points, (a, b, c)):
        point.set_data([x], [f(x)])

    # Update text annotation
    text.set_text(
        f"Iterations: {len(a_gray.get_data()[0])}\na = {a:.5f}\nb = {b:.5f}\nc = {c:.5f}\nf(c) = {fc:.5f}"
    )

    # Dynamically adjust the axes to fit the current interval with a minimum margin
    margin = max(min_margin, (b - a) * 0.7)  # Increase margin to ensure separation
    ax.set_xlim(min(a, b) - margin, max(a, b) + margin)

    # Include all y-values from gray trajectories to set y-axis limits
    all_y = np.concatenate(
        [a_gray.get_data()[1], b_gray.get_data()[1], c_gray.get_data()[1]]
    )
    y_margin = 1.5 * max(
        abs(f(a)), abs(f(b))
    )  # Use the absolute value of the maximum multiplied by a coefficient
    y_min = min(all_y) - y_margin  # Calculate the minimum value of the y-axis
    y_max = max(all_y) + y_margin  # Calculate the maximum value of the y-axis

    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Bisection Method Iteration {len(a_gray.get_data()[0])}")


def main():
    print("Bisection Method Dynamic Visualization")
    print("Choose the interval to iterate for the root:")
    print("1: Find the root in the interval [0.0, 1.0]")
    print("2: Find the root in the interval [1.0, 2.0]")
    print("3: Custom interval, enter the left and right endpoints")

    choice = input("Please enter your choice (1/2/3):")

    if choice == "1":
        a, b = 0.0, 1.0
    elif choice == "2":
        a, b = 1.0, 2.0
    elif choice == "3":
        try:
            a = float(input("Please enter the left endpoint a: "))
            b = float(input("Please enter the right endpoint b: "))
        except ValueError:
            print("Invalid input. Please ensure you enter numbers. Exiting program.")
            sys.exit(1)
    else:
        print("Invalid choice. Exiting program.")
        sys.exit(1)

    tol, max_iter = 1e-4, 100
    try:
        iterations = bisection_method(a, b, tol, max_iter)
    except ValueError as e:
        print(e)
        sys.exit(1)

    if not iterations:
        print("No iteration records. Exiting program.")
        sys.exit(1)

    final_a, final_b, _, _, _, _ = iterations[
        -1
    ]  # Use the results of the last iteration
    x_min, x_max = (
        min(final_a, final_b) - 1.5,  # Increase the margin of the x-axis
        max(final_a, final_b) + 1.5,
    )
    # Dynamically adjust the margin of the y-axis
    y_margin = 1.5 * max(
        abs(f(final_a)), abs(f(final_b))
    )  # Use the absolute value of the maximum multiplied by a coefficient
    y_min = (
        min(f(final_a), f(final_b)) - y_margin
    )  # Calculate the minimum value of the y-axis
    y_max = (
        max(f(final_a), f(final_b)) + y_margin
    )  # Calculate the maximum value of the y-axis

    # Prepare the plot
    fig, ax = plt.subplots()
    init_plot(ax, x_min, x_max, y_min, y_max)

    # Initialize gray trajectory lines for previous points
    gray_lines = (
        ax.plot([], [], color="gray")[0],
        ax.plot([], [], color="gray")[0],
        ax.plot([], [], color="gray")[0],
    )
    current_points = (
        ax.plot([], [], "ro")[0],
        ax.plot([], [], "bo")[0],
        ax.plot([], [], "go")[0],
    )

    ax.legend()
    text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=iterations,
        fargs=(gray_lines, current_points, ax, text),
        init_func=lambda: None,
        blit=False,
        repeat=False,
        interval=1000,
    )
    plt.show()

    # Print the final result
    root = (final_a + final_b) / 2
    print(f"\nConverged to root: {root:.14f}, after {len(iterations)} iterations.")


if __name__ == "__main__":
    main()
