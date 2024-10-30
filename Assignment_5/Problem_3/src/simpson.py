import numpy as np
import math
import matplotlib.pyplot as plt

# Constants
Z = 14  # Effective nuclear charge for Si
n = 3  # Principal quantum number for 3s orbital
r_max = 40  # Upper limit of integration in atomic units
r0 = 0.0005  # Parameter for non-uniform grid in atomic units
e = np.e  # Base of natural logarithm


def integrand(r, Z=14, n=3):
    """
    Integrand |R_3s(r)|^2 * r^2.

    Parameters:
    r : float or np.ndarray
        Radial distance in atomic units.
    Z : int
        Effective nuclear charge.
    n : int
        Principal quantum number.

    Returns:
    integrand_value : float or np.ndarray
        Value of the integrand at r.
    """
    rho = (2 * Z * r) / n
    pre_factor = (1 / (9 * np.sqrt(3))) * (6 - 6 * rho + rho**2) * Z**1.5
    exponent = np.exp(-rho / 2)
    R = pre_factor * exponent
    return np.abs(R) ** 2 * r**2


def simpsons_rule(r, f):
    """
    Compute the integral using Simpson's rule.

    Parameters:
    r : np.ndarray
        Grid points.
    f : np.ndarray
        Function values at grid points.

    Returns:
    integral : float
        Approximated integral value.
    """
    N = len(r)
    if N < 3:
        raise ValueError("Simpson's rule requires at least 3 points.")
    if N % 2 == 0:
        # If N is even, Simpson's rule requires an odd number of points
        # Remove the last point
        r = r[:-1]
        f = f[:-1]
        N -= 1

    h = (r[-1] - r[0]) / (N - 1)
    S = f[0] + f[-1]

    # Sum over odd indices
    S += 4 * np.sum(f[1:-1:2])

    # Sum over even indices
    S += 2 * np.sum(f[2:-2:2])

    integral = (h / 3) * S
    return integral


def compute_integral_equal_spacing(
    Z=14, n=3, r_max=40, N_values=[11, 21, 51, 101, 201]
):
    results = {}
    for N in N_values:
        r = np.linspace(0, r_max, N)
        f = integrand(r, Z, n)
        integral = simpsons_rule(r, f)
        results[N] = integral
    return results


def compute_integral_exp_spacing(
    Z=14, n=3, r_max=40, r0=0.0005, N_values=[11, 21, 51, 101, 201]
):
    results = {}
    for N in N_values:
        # Compute t_max
        t_max = np.log((r_max / r0) + 1)
        t = np.linspace(0, t_max, N)
        r = r0 * (np.exp(t) - 1)
        f = integrand(r, Z, n)
        frator = r0 * np.exp(t)
        integral = simpsons_rule(t, f * frator)
        results[N] = integral
    return results


# Theoretical expected value by Mathematica
numerator = 242794431087524801
denominator = 6561 * math.exp(1120 / 3)
err = numerator / denominator
theoretical_value = 1 - err

if __name__ == "__main__":
    print(f"(Theoretical Expected : 1 - {err:.1e})")
    N_values = [3, 5, 7, 9, 11, 21, 51, 101, 201, 501, 1001, 10001]
    # Equal spacing grids
    equal_spacing_results = compute_integral_equal_spacing(N_values=N_values)
    print("Equal Spacing Grid Results:")
    for N, integral in equal_spacing_results.items():
        print(f"N = {N}: Integral = {integral:.6f}")

    # Non-uniform spacing grids
    exp_spacing_results = compute_integral_exp_spacing(N_values=N_values)
    print("\nExpotential Spacing Grid Results:")
    for N, integral in exp_spacing_results.items():
        print(f"N = {N}: Integral = {integral:.6f}")

    # Plot settings
    selected_N = [11, 51, 201, 1001]  # Selected N values for plotting
    colors = {"Equal": "blue", "Exponential": "red"}

    # Generate a fine grid for plotting the integrand
    r_plot = np.linspace(0, r_max, 1000)
    f_plot = integrand(r_plot, Z, n)

    # Create subplots
    fig, axs = plt.subplots(
        len(selected_N), 1, figsize=(8, 4 * len(selected_N)), sharex=True
    )

    if len(selected_N) == 1:
        axs = [axs]  # Ensure axs is iterable

    for idx, N in enumerate(selected_N):
        ax = axs[idx]
        ax.plot(r_plot, f_plot, label="Integrand", color="gray", linewidth=1)

        # Equal spacing grid
        r_equal = np.linspace(0, r_max, N)
        f_equal = integrand(r_equal, Z, n)
        ax.scatter(
            r_equal, f_equal, color=colors["Equal"], label="Equal Spacing", zorder=5
        )

        # Exponential grid
        t_max = np.log((r_max / r0) + 1)
        t = np.linspace(0, t_max, N)
        r_exp = r0 * (np.exp(t) - 1)
        f_exp = integrand(r_exp, Z, n)
        ax.scatter(
            r_exp,
            f_exp,
            color=colors["Exponential"],
            label="Exponential Spacing",
            zorder=5,
        )

        ax.set_title(f"Grid Points for N = {N}")
        ax.set_ylabel("Integrand")
        ax.legend()

    axs[-1].set_xlabel("r (atomic units)")
    plt.tight_layout()
    plt.show()
