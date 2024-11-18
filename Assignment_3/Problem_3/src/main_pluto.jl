# main_pluto.jl (Pluto adapted)
# Main script for solving the 1D Schrödinger equation using Gaussian basis functions
# with additional utility modules.
#
# Author: Gilbert Young
# Date: 2024/10/08
#
# This Pluto notebook allows the user to select different potential functions,
# solve the Schrödinger equation, and visualize the results interactively.

using PlutoUI
using LinearAlgebra
using Plots

# Import custom modules
include("methods.jl")
include("utils.jl")
include("interaction.jl")
using .Methods
using .Utils
using .Interaction

# Interactive selection of potential functions
@bind choice Dropdown(["1", "2", "3", "4"], "Select potential function:")
@bind num_levels Slider(1:10, show_value=true, label="Number of energy levels to compute")

# Get parameters based on selected potential function
(N, v, s, potential_gaussian_integral_list, potential_params_list, potential_name_list) = get_parameters(choice)

# Calculate Hamiltonian and overlap matrices
H, S = build_matrices(N, v, s, potential_gaussian_integral_list[1], potential_params_list[1])

# Solve the Schrödinger equation for the chosen potential
energies, states = solve_schrodinger(H, S, num_levels)

# Display energy levels
md"""
### Lowest $(num_levels) energy levels for $potential_name_list[1]:
"""
for (j, E) in enumerate(energies)
    @bind _ println("Energy Level $(j): E = $(E)")
end

# Plot the wave functions
x_vals = range(-5, 5, length=200)

# Compute wavefunctions in parallel
wavefunctions = [zeros(Float64, length(x_vals)) for _ in 1:num_levels]

@threads for n in 1:num_levels
    ψ_n = zeros(Float64, length(x_vals))

    # Construct the wave function as a linear combination of Gaussian basis functions
    for k in 1:N
        ψ_n .+= states[k, n] * sqrt(v[k] / π) .* exp.(-v[k] .* (x_vals .- s[k]) .^ 2)
    end

    # Normalize the wave function
    wavefunctions[n] = normalize_wavefunction(x_vals, ψ_n)
end

# Display plot control and plot the wave functions
@bind plot_choice Toggle(label="Show wave functions plot")

if plot_choice
    plot_wavefunctions(x_vals, wavefunctions, num_levels, potential_name_list[1], potential_params_list[1])
end
