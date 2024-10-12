# utils.jl
# Utility functions for the Schrödinger equation solver.
#
# Author: Gilbert Young
# Date: 2024/10/08

module Utils

using LinearAlgebra
using ..Methods

export check_positive_v, trapezoidal_integration, get_parameters, get_potential_function, read_number, read_int, normalize_wavefunction

@doc raw"""
    check_positive_v(v)

Checks if `v` is positive.

# Arguments
- `v`: Value to check.

# Throws
- `Error` if `v <= 0`.
"""
function check_positive_v(v::Float64)
    if v <= 0
        error("Error: v must be greater than 0. Please enter a positive value for v.")
    end
end

@doc raw"""
    trapezoidal_integration(x, y)

Performs trapezoidal integration of `y` with respect to `x`.

# Arguments
- `x`: Vector of x values.
- `y`: Vector of y values.

# Returns
- `integral`: The result of the integration.
"""
function trapezoidal_integration(x::AbstractVector{Float64}, y::AbstractVector{Float64})
    integral = sum((y[1:end-1] + y[2:end]) .* diff(x)) / 2
    return integral
end

@doc raw"""
    read_number(prompt::String, default::Float64)

Reads a floating-point number from the user with a prompt. If the user presses enter without input, returns the default value.

# Arguments
- `prompt`: The prompt message to display.
- `default`: The default value to return if no input is provided.

# Returns
- The user-entered number or the default value.
"""
function read_number(prompt::String, default::Float64)
    print("$prompt [Default: $default] ")
    input = readline()
    if isempty(input)
        return default
    else
        try
            return parse(Float64, input)
        catch
            println("\nInvalid input. Using default value: $default")
            return default
        end
    end
end

@doc raw"""
    read_int(prompt::String, default::Int)

Reads an integer from the user with a prompt. If the user presses enter without input, returns the default value.

# Arguments
- `prompt`: The prompt message to display.
- `default`: The default value to return if no input is provided.

# Returns
- The user-entered integer or the default value.
"""
function read_int(prompt::String, default::Int)
    print("$prompt [Default: $default] ")
    input = readline()
    if isempty(input)
        return default
    else
        try
            return parse(Int, input)
        catch
            println("\nInvalid input. Using default value: $default")
            return default
        end
    end
end

@doc raw"""
    normalize_wavefunction(x_vals::AbstractVector{Float64}, ψ::AbstractVector{Float64})::Vector{Float64}

Normalizes the wave function `ψ` based on the integration over `x_vals`.

# Arguments
- `x_vals`: Vector of x values.
- `ψ`: Wave function to normalize.

# Returns
- Normalized wave function.
"""
function normalize_wavefunction(x_vals::AbstractVector{Float64}, ψ::AbstractVector{Float64})::Vector{Float64}
    norm_factor = sqrt(trapezoidal_integration(x_vals, ψ .^ 2))
    return ψ / norm_factor
end

@doc raw"""
    get_parameters(choice::String)

Retrieves parameters based on the user's selection.

# Arguments
- `choice`: User selection for the potential type.

# Returns
- `(N, v, s, potential_gaussian_integral, potential_params, potential_name, num_levels)`: Parameters for solving the Schrödinger equation.
"""
function get_parameters(choice::String)
    num_levels_default = 3
    if choice in ["1", "2", "4"]
        N = read_int("Enter the number of basis functions N:", 40)

        println("Select parameter setting:")
        println("1. Fixed v, varying s")
        println("2. Fixed s, varying v")
        print("Enter your choice: ")
        param_choice = readline()

        v, s = Float64[], Float64[]
        if param_choice == "1"
            v_value = read_number("Enter the fixed value of v:", 0.5)
            check_positive_v(v_value)
            v = fill(v_value, N)

            println("Suggested range for s is from -$(N/4) to $(N/4)")
            s_start = read_number("Enter the starting value of s:", -N / 4)
            s_end = read_number("Enter the ending value of s:", N / 4)
            s = collect(range(s_start, stop=s_end, length=N))
        elseif param_choice == "2"
            s_value = read_number("Enter the fixed value of s:", 0.0)
            s = fill(s_value, N)

            v_start = read_number("Enter the starting value of v:", 0.1)
            check_positive_v(v_start)

            v_end = read_number("Enter the ending value of v:", 1.0)
            check_positive_v(v_end)

            v = collect(range(v_start, stop=v_end, length=N))
        else
            println("Invalid selection, returning to main menu.")
            return 0, Float64[], Float64[], nothing, Float64[], "", 0
        end

        num_levels = read_int("Enter the number of energy levels to compute:", num_levels_default)

        potential_gaussian_integral_list, potential_name_list, potential_params_list = get_potential_function(choice)
        return N, v, s, potential_gaussian_integral_list, potential_params_list, potential_name_list, num_levels
    elseif choice == "3"
        N = 40
        v = fill(0.5, N)
        s = collect(range(-10.0, stop=10.0, length=N))
        num_levels = num_levels_default
        potential_gaussian_integral_list, potential_name_list, potential_params_list = get_potential_function("3")
        return N, v, s, potential_gaussian_integral_list, potential_params_list, potential_name_list, num_levels
    else
        println("Invalid selection, returning to main menu.")
        return 0, Float64[], Float64[], nothing, Float64[], "", 0
    end
end

@doc raw"""
    get_potential_function(choice::String)

Selects the potential function based on user's choice.

# Arguments
- `choice`: User selection for the potential type.

# Returns
- `[(potential_gaussian_integral, potential_name, potential_params)]`: Potential function and parameters.
"""
function get_potential_function(choice::String)
    if choice == "1"
        potential_gaussian_integral = (v1, s1, v2, s2, params...) -> potential_integral_xn(v1, s1, v2, s2, 2)
        potential_name = "V(x) = x^2"
        potential_params = [0.0, 0.0, 1.0, 0.0, 0.0]  # Coefficients from x^4 to x^0
        return [potential_gaussian_integral], [potential_name], [potential_params]
    elseif choice == "2"
        potential_gaussian_integral = (v1, s1, v2, s2, params...) -> potential_integral_xn(v1, s1, v2, s2, 4) - potential_integral_xn(v1, s1, v2, s2, 2)
        potential_name = "V(x) = x^4 - x^2"
        potential_params = [1.0, 0.0, -1.0, 0.0, 0.0]
        return [potential_gaussian_integral], [potential_name], [potential_params]
    elseif choice == "3"
        potential_gaussian_integral_list = [
            (v1, s1, v2, s2, params...) -> potential_integral_xn(v1, s1, v2, s2, 2),
            (v1, s1, v2, s2, params...) -> potential_integral_xn(v1, s1, v2, s2, 4) - potential_integral_xn(v1, s1, v2, s2, 2)
        ]
        potential_name_list = ["V(x) = x^2", "V(x) = x^4 - x^2"]
        potential_params_list = [[0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, -1.0, 0.0, 0.0]]
        return potential_gaussian_integral_list, potential_name_list, potential_params_list
    elseif choice == "4"
        println("Enter the coefficients of the polynomial potential (degree up to 4).")
        println("Format: c4 c3 c2 c1 c0 (separated by spaces) [Default: 1 0 -1 0 0]")
        print("Enter coefficients: ")
        potential_params_input = readline()
        if isempty(potential_params_input)
            potential_params = [1.0, 0.0, -1.0, 0.0, 0.0]
        else
            try
                potential_params = parse.(Float64, split(potential_params_input))
                if length(potential_params) != 5
                    error("Please enter exactly 5 coefficients.")
                end
            catch
                println("Invalid input. Using default coefficients: 1 0 -1 0 0")
                potential_params = [1.0, 0.0, -1.0, 0.0, 0.0]
            end
        end
        potential_gaussian_integral = (v1, s1, v2, s2, potential_params...) -> sum(potential_params[i] * potential_integral_xn(v1, s1, v2, s2, 4 - (i - 1)) for i in eachindex(potential_params))
        terms = [c != 0 ? "$(c)x^$(4 - (i - 1))" : "" for (i, c) in enumerate(potential_params)]
        potential_name = "V(x) = " * join(filter(!isempty, terms), " + ")
        return [potential_gaussian_integral], [potential_name], [potential_params]
    else
        error("\nInvalid potential function selection.")
    end
end

end  # module Utils
