"""
@Author: Gilbert Young
@Time: 2024/09/20 20:41
@File_name: schrodinger.py
@IDE: Vscode
@Formatter: Black
@Description: This script solves the Schrödinger equation for a particle in a finite potential well using the finite difference method.
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

# Constants
hbar = 1.0545718e-34  # Reduced Planck's constant (J·s)
m = 9.10938356e-31  # Electron mass (kg)
eV_to_J = 1.60218e-19  # eV to Joules conversion
V0 = 10 * eV_to_J  # Potential well depth (J)
a = 0.2e-9  # Well width (m)
N = 1000  # Grid points
L = 10e-9  # Region range (m)
dx = 2 * L / (N - 1)  # Step size

# Grid points (m)
x = np.linspace(-L, L, N)

# Potential function (J)
V = np.zeros(N)
V[np.abs(x) <= a] = -V0

# Hamiltonian matrix (J)
H = np.zeros((N, N), dtype=np.float64)
for i in range(1, N - 1):
    H[i, i] = -2.0
    H[i, i + 1] = 1.0
    H[i, i - 1] = 1.0

H *= -(hbar**2) / (2 * m * dx**2)
H += np.diag(V)

# Eigenvalues and eigenvectors
E, psi = scipy.linalg.eigh(H)

# Bound state energy levels and wavefunctions
E_bound = E[E < 0][1:]  # Exclude the first solution
psi_bound = psi[:, E < 0][:, 1:]

# Energy levels in eV
E_bound_eV = E_bound / eV_to_J
print("Bound state energy levels (eV):", E_bound_eV)

# Plot wavefunctions
plt.figure(figsize=(10, 6))
for i in range(len(E_bound_eV)):
    mask = (x >= -3e-9) & (x <= 3e-9)
    plt.plot(
        x[mask] * 1e9, psi_bound[mask, i], label=f"n={i+1}, E={E_bound_eV[i]:.2f} eV"
    )

plt.xlabel("x (nm)")
plt.ylabel("ψ(x)")
plt.title("Wavefunctions in a Finite Potential Well")
plt.legend()
plt.grid()
plt.show()
