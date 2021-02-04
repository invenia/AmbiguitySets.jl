module AmbiguitySets

using Distributions
using LinearAlgebra


"""
    AmbiguitySet <: Sampleable

Defines the ambiguity related to a contained distribution, often used in
Robust Optimization (RO) and Distributionally Robust Optimization (DRO) problems.
It represents a bounded infinite set of distributions.

For more information on how `AmbiguitySet`s are used for RO and DRO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
abstract type AmbiguitySet{T<:Real, D<:Sampleable{Multivariate, Continuous}} <: Sampleable{Multivariate, Continuous} end

Base.rand(s::AmbiguitySet) = rand(distribution(s))

# NOTE: Technically we could probably implement `rand` for `Bertsimas` and `BenTal` by
# uniformly augmenting the samples from the underlying MvNormal with the uncertainty
# estimate. I don't think anything like that would be possible for Delague though.

"""
    Bertsimas <: AmbiguitySet

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uncertainty around mean. (default: std(d) / 5)
- `Γ::Float64`: Number of assets in worst case. (default: 0.1 * length(d))

For more information on how Bertsimas' uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct Bertsimas{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    d::D
    Δ::Vector{T}
    Γ::T

    # Inner constructor for validating arguments
    function Bertsimas{T, D}(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:Sampleable}
        length(d) == length(Δ) || throw(ArgumentError(
            "Distribution ($(length(d))) and Δ ($(length(d))) are not the same length"
        ))
        all(>=(0), Δ) || throw(ArgumentError("All uncertainty deltas must be >= 0"))
        return new{T, D}(d, Δ, Γ)
    end
end

# Default outer constructor
Bertsimas(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:Sampleable} = Bertsimas{T, D}(d, Δ, Γ)

# Kwarg constructor with defaults
function Bertsimas(
    d::AbstractMvNormal;
    Δ=default_bertsimas_delta(d),
    Γ=default_bertsimas_budget(d),
)
    return Bertsimas(d, Δ, Γ)
end

# Not sure about these defaults, but it seems like something we should support?
default_bertsimas_delta(d::AbstractMvNormal) = sqrt.(var(d)) ./ 5
default_bertsimas_budget(d::Sampleable{Multivariate}) = 0.1 * length(d)

distribution(s::Bertsimas) = s.d

"""
    BenTal <: AmbiguitySet

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uniform uncertainty around mean. (default: std(dist) / 5)

For more information on how BenTal uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct BenTal{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    d::D
    Δ::T

    # Inner constructor for validating arguments
    function BenTal{T, D}(d::D, Δ::T) where {T<:Real, D<:Sampleable}
        Δ >= 0 || throw(ArgumentError("Uncertainty delta must be >= 0"))
        return new{T, D}(d, Δ)
    end
end

# Default outer constructor
BenTal(d::D, Δ::T) where {T<:Real, D<:Sampleable} = BenTal{T, D}(d, Δ)

# Kwarg constructor with default delta value
BenTal(d::AbstractMvNormal; Δ=default_ben_tal_delta(d)) = BenTal(d, Δ)

default_ben_tal_delta(d::AbstractMvNormal) = first(sqrt.(var(d)) ./ 5)

distribution(s::BenTal) = s.d

"""
    Delague <: AmbiguitySet

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0). (default: std(dist) / 5)
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1). (default: 3.0)
- `coefficients::Vector{Float64}`: Piece-wise utility coeficients (default [1.0]).
- `intercepts::Vector{Float64}`: Piece-wise utility intercepts (default [0.0]).

For more information on how BenTal uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct Delague{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    d::D
    γ1::T
    γ2::T
    coefficients::Vector{T}
    intercepts::Vector{T}

    # Inner constructor for validating arguments
    function Delague{T, D}(
        d::D, γ1::T, γ2::T, coefficients::Vector{T}, intercepts::Vector{T}
    ) where {T<:Real, D<:Sampleable}
        length(coefficients) == length(intercepts) || throw(ArgumentError(
            "Length of coefficients ($(length(coefficients))) and intercepts " *
            "($(length(intercepts))) do not match"
        ))
        γ1 >= 0 || throw(ArgumentError("γ1 must be >= 0"))
        γ2 >= 1 || throw(ArgumentError("γ2 must be >= 1"))
        return new{T, D}(d, γ1, γ2, coefficients, intercepts)
    end
end

# Default outer constructor
function Delague(
    d::D, γ1::T, γ2::T, coefficients::Vector{T}, intercepts::Vector{T}
) where {T<:Real, D<:Sampleable}
    Delague{T, D}(d, γ1, γ2, coefficients, intercepts)
end

# Kwarg constructor with defaults
function Delague(
    d::AbstractMvNormal;
    γ1=default_delague_γ1(d), γ2=3.0, coefficients=[1.0], intercepts=[0.0],
)
    return Delague(d, γ1, γ2, coefficients, intercepts)
end

default_delague_γ1(d::AbstractMvNormal) = first(sqrt.(diag(cov(d))) ./ 5)

distribution(s::Delague) = s.d

# NOTE: The Betina formulation doesn't use ambiguity sets?

end
