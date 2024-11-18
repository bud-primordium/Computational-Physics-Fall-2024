# methods.jl
# Methods for solving the 1D Schrödinger equation using Gaussian basis functions
#
# Author: Gilbert Young
# Date: 2024/10/08
#
# This module provides functions to compute overlap, kinetic energy, and potential energy integrals
# for Gaussian basis functions and to solve the 1D Schrödinger equation using the variational method.

module Methods

using LinearAlgebra
using Base.Threads  # For parallel computing

export overlap_integral, kinetic_integral, potential_integral_xn
export build_matrices, solve_schrodinger

@doc raw"""
    overlap_integral(v1, s1, v2, s2)

Computes the overlap integral between two Gaussian basis functions with parameters `(v1, s1)` and `(v2, s2)`.

The overlap integral is given by:

```math
S_{ij} = \frac{\sqrt{v_1 v_2}}{\sqrt{\pi (v_1 + v_2)}} e^{- \frac{v_1 v_2 (s_1 - s_2)^2}{v_1 + v_2}}
```

# Arguments

- `v1`: Width parameter of the first Gaussian basis function.
- `s1`: Center of the first Gaussian basis function.
- `v2`: Width parameter of the second Gaussian basis function.
- `s2`: Center of the second Gaussian basis function.

# Returns

- `S_ij`: The overlap integral value.
"""
function overlap_integral(v1::Real, s1::Real, v2::Real, s2::Real)::Float64
    vp = v1 + v2
    exponent = -((v1 * v2 * (s1 - s2)^2) / vp)
    prefactor = sqrt(v1 * v2) / (sqrt(π) * sqrt(vp))
    S_ij = prefactor * exp(exponent)
    return S_ij
end

@doc raw"""
    kinetic_integral(v1, s1, v2, s2)

Computes the kinetic energy integral between two Gaussian basis functions.

The kinetic energy integral is given by:

```math
T_{ij} = \frac{v_1^{3/2} v_2^{3/2} \left( v_1 + v_2 - 2 v_1 v_2 (s_1 - s_2)^2 \right)}{\sqrt{\pi} (v_1 + v_2)^{5/2}} e^{- \frac{v_1 v_2 (s_1 - s_2)^2}{v_1 + v_2}}
```

# Arguments

- `v1`: Width parameter of the first Gaussian basis function.
- `s1`: Center of the first Gaussian basis function.
- `v2`: Width parameter of the second Gaussian basis function.
- `s2`: Center of the second Gaussian basis function.

# Returns

- `T_ij`: The kinetic energy integral value.
"""
function kinetic_integral(v1::Real, s1::Real, v2::Real, s2::Real)::Float64
    vp = v1 + v2
    exponent = -((v1 * v2 * (s1 - s2)^2) / vp)
    numerator = v1 + v2 - 2 * v1 * v2 * (s1 - s2)^2
    prefactor = v1^(1.5) * v2^(1.5) / (sqrt(π) * vp^(2.5))
    T_ij = prefactor * numerator * exp(exponent)
    return T_ij
end

@doc raw"""
    potential_integral_xn(v1, s1, v2, s2, n)

Computes the potential energy integral for \( V(x) = x^n \) between two Gaussian basis functions.

# Arguments

- `v1`, `s1`: Parameters of the first Gaussian basis function.
- `v2`, `s2`: Parameters of the second Gaussian basis function.
- `n`: The power of x in the potential function (integer from 0 to 4).

# Returns

- `V_ij`: The potential energy integral value.
"""
function potential_integral_xn(v1::Real, s1::Real, v2::Real, s2::Real, n::Int)::Float64
    if n < 0 || n > 4
        error("n should be an integer between 0 and 4")
    end

    vp = v1 + v2
    exponent = -((v1 * v2 * (s1 - s2)^2) / vp)
    S_ij = sqrt(v1 * v2) / (sqrt(π) * sqrt(vp)) * exp(exponent)

    μ = (v1 * s1 + v2 * s2) / vp
    σ2 = 1 / (2 * (v1 + v2))

    # Compute the nth moment M_n(μ, σ2)
    if n == 0
        moment = 1.0
    elseif n == 1
        moment = μ
    elseif n == 2
        moment = μ^2 + σ2
    elseif n == 3
        moment = μ^3 + 3 * μ * σ2
    elseif n == 4
        moment = μ^4 + 6 * μ^2 * σ2 + 3 * σ2^2
    end

    V_ij = S_ij * moment
    return V_ij
end

@doc raw"""
    build_matrices(N, v, s, potential_gaussian_integral, potential_params)

Builds the Hamiltonian matrix `H` and overlap matrix `S` for the variational method.

# Arguments

- `N`: Number of basis functions.
- `v`: Width parameters of the Gaussian basis functions.
- `s`: Centers of the Gaussian basis functions.
- `potential_gaussian_integral`: Function to compute the potential energy integral.
- `potential_params`: Additional parameters for the potential function.

# Returns

- `(H, S)`: The Hamiltonian and overlap matrices.
"""
function build_matrices(N::Int, v::Vector{Float64}, s::Vector{Float64}, potential_gaussian_integral, potential_params)
    H = zeros(N, N)
    S = zeros(N, N)
    @threads for i in 1:N
        for j in i:N  # Utilize symmetry, compute only upper triangle
            S_ij = overlap_integral(v[i], s[i], v[j], s[j])
            T_ij = kinetic_integral(v[i], s[i], v[j], s[j])
            V_ij = potential_gaussian_integral(v[i], s[i], v[j], s[j], potential_params...)
            H_ij = T_ij + V_ij
            H[i, j] = H_ij
            S[i, j] = S_ij
            if i != j
                H[j, i] = H_ij
                S[j, i] = S_ij
            end
        end
    end
    return H, S
end

@doc raw"""
    solve_schrodinger(H, S, num_levels)

Solves the generalized eigenvalue problem for the Hamiltonian `H` and overlap matrix `S`.

# Arguments

- `H`: Hamiltonian matrix.
- `S`: Overlap matrix.
- `num_levels`: Number of energy levels to compute.

# Returns

- `(energies, states)`: The lowest `num_levels` eigenvalues and eigenvectors.
"""
function solve_schrodinger(H::Matrix{Float64}, S::Matrix{Float64}, num_levels::Int)
    E_values, E_vectors = eigen(H, S)
    idx = sortperm(real(E_values))
    energies = real(E_values[idx[1:num_levels]])
    states = E_vectors[:, idx[1:num_levels]]
    return energies, states
end

end  # module Methods