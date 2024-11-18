# main.jl
# Main script for solving the 1D Schrödinger equation using Gaussian basis functions
# with additional utility modules.
#
# Author: Gilbert Young
# Date: 2024/10/08
#
# This script provides the user interface to select different potential functions,
# solve the Schrödinger equation, and visualize the results.
module SchrödingerSolver

include("methods.jl")
include("utils.jl")
include("interaction.jl")

using .Methods
using .Utils
using .Interaction

using LinearAlgebra
using Plots
using Base.Threads

function main()
    print("Number of threads available: ", Threads.nthreads())
    while true
        display_menu()
        # Prompt for main menu selection
        print("Enter your choice: ")
        choice = readline()

        if choice in ["1", "2", "3", "4"]
            N, v, s, potential_gaussian_integral_list, potential_params_list, potential_name_list, num_levels = get_parameters(choice)
            if N == 0
                continue
            end

            # Loop over potential functions and corresponding parameters/names
            for i in eachindex(potential_gaussian_integral_list)
                potential_gaussian_integral = potential_gaussian_integral_list[i]
                potential_params = potential_params_list[i]
                potential_name = potential_name_list[i]

                # Build Hamiltonian and overlap matrices
                H, S = build_matrices(N, v, s, potential_gaussian_integral, potential_params)

                # Solve the Schrödinger equation
                energies, states = solve_schrodinger(H, S, num_levels)

                # Output results
                println("\nLowest $(num_levels) energy levels for $potential_name:")
                for (j, E) in enumerate(energies)
                    println("Energy Level $(j): E = $(E)")
                end

                # Ask whether to plot
                print("Do you want to plot the wave functions for $potential_name? (y/n) ")
                plot_choice = readline()
                if plot_choice in ["y", "Y"]
                    x_vals = range(-5, 5, length=200)

                    # Parallel computation of wave functions
                    wavefunctions = Vector{Vector{Float64}}(undef, num_levels)

                    @threads for n in 1:num_levels
                        ψ_n = zeros(Float64, length(x_vals))

                        # Construct the wave function as a linear combination of Gaussian basis functions
                        for k in 1:N
                            ψ_n .+= states[k, n] * sqrt(v[k] / π) .* exp.(-v[k] .* (x_vals .- s[k]) .^ 2)
                        end

                        # Normalize the wave function using the utility function
                        wavefunctions[n] = normalize_wavefunction(x_vals, ψ_n)
                    end

                    # Plot the precomputed wave functions
                    plot_wavefunctions(x_vals, wavefunctions, num_levels, potential_name, potential_params)
                end
            end
        elseif choice in ["q", "Q"]
            print("Program exited.")
            break
        else
            print("Invalid selection, please try again.\n")
        end
    end
end

# Run the main function if the script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module SchrödingerSolver
