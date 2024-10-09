# MyInteraction.jl
# Interaction functions for plotting and displaying the Schrödinger equation solver results.
#
# Author: Gilbert Young
# Date: 2024/10/08

module MyInteraction

using Plots
using MyUtils

export display_menu, plot_wavefunctions

@doc raw"""
    display_menu()

Displays the main menu options to the user.
"""
function display_menu()
    println("\nMain Menu:")
    println("1. Custom parameters for V(x) = x^2")
    println("2. Custom parameters for V(x) = x^4 - x^2")
    println("3. Use default parameters to compute both problems")
    println("4. Custom polynomial potential function up to degree 4")
    println("q. Quit")
end

@doc raw"""
    plot_wavefunctions(
        x_vals::AbstractVector{Float64},
        wavefunctions::Vector{Vector{Float64}},
        num_levels::Int,
        potential_name::String,
        potential_params::Vector{Float64}
    )

Plots the wave functions.

# Arguments
- `x_vals`: Values of `x`.
- `wavefunctions`: Precomputed and normalized wave functions.
- `num_levels`: Number of energy levels to plot.
- `potential_name`: The name of the potential function.
- `potential_params`: Coefficients for the polynomial potential function.
"""
function plot_wavefunctions(
    x_vals::AbstractVector{Float64},
    wavefunctions::Vector{Vector{Float64}},
    num_levels::Int,
    potential_name::String,
    potential_params::Vector{Float64}
)
    # Initialize the primary plot for wave functions on the left y-axis
    p = plot(title="Wave Functions and Potential\n($potential_name)", xlabel="x", ylabel="Ψ(x)", legend=:topleft)

    # Batch plot wave functions
    for n in 1:num_levels
        plot!(p, x_vals, wavefunctions[n], label="Ψ$n")
    end

    # Define the potential function based on the given parameters
    V_function = x -> sum(potential_params[i] * x^(4 - (i - 1)) for i in eachindex(potential_params))

    # Calculate the potential values for plotting
    V_vals = [V_function(x) for x in x_vals]

    # Use `twinx()` to add a second y-axis for the potential
    p_right = twinx()
    plot!(p_right, x_vals, V_vals, label="Potential V(x)", linewidth=2, linestyle=:solid, color=:black, ylabel="V(x)", legend=:topright)

    # Display the final plot with dual y-axes
    display(p)
end

end  # module Interaction
