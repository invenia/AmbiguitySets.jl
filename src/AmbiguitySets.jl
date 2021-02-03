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
    Bertsimas <: UncertaintySet

Atributes:
- `dist::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `uncertainty::Array{Float64,1}` (latex notation ``\\Delta``): Uncertainty around mean. (default: std(dist) / 5)
- `budget::Float64` (latex notation ``\\Gamma``): Number of assets in worst case. (default: 0.1 * length(dist))

For more information on how Bertsimas' uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct Bertsimas{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    dist::D
    uncertainty::Vector{T}
    budget::T
end

# Not sure about these defaults, but it seems like something we should support?
default_bertsimas_uncertainty(d::AbstractMvNormal) = sqrt.(diag(cov(d))) ./ 5
default_bertsimas_budget(d::Sampleable{Multivariate}) = 0.1 * length(d)

function Bertsimas(
    d::AbstractMvNormal;
    uncertainty=default_bertsimas_uncertainty(d),
    budget=default_bertsimas_budget(d),
)
    return Bertsimas(d, uncertainty, budget)
end

distribution(s::Bertsimas) = s.dist

"""
    BenTal <: UncertaintySet

Atributes:
- `dist::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `uncertainty::Array{Float64,1}` (latex notation ``\\Delta``): Uniform uncertainty around mean. (default: std(dist) / 5)

For more information on how BenTal uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct BenTal{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    dist::D
    uncertainty::T
end

default_ben_tal_uncertainty(d::AbstractMvNormal) = first(sqrt.(diag(cov(d))) ./ 5)

function BenTal(d::AbstractMvNormal; uncertainty=default_ben_tal_uncertainty(d))
    return BenTal(d, uncertainty)
end

distribution(s::BenTal) = s.dist

"""
    Delague <: AmbiguitySet

Atributes:
- `dist::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0). (default: std(dist) / 5)
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1). (default: 3.0)
- `coefficients::Vector{Float64}`: Piece-wise utility coeficients (default [1.0]).
- `intercepts::Vector{Float64}`: Piece-wise utility intercepts (default [0.0]).

For more information on how BenTal uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct Delague{T<:Real, D<:Sampleable} <: AmbiguitySet{T, D}
    dist::D
    γ1::T
    γ2::T
    coefficients::Vector{T}
    intercepts::Vector{T}
end

default_delague_γ1(d::AbstractMvNormal) = first(sqrt.(diag(cov(d))) ./ 5)

function Delague(
    d::AbstractMvNormal;
    γ1=default_delague_γ1(d), γ2=3.0, coefficients=[1.0], intercepts=[0.0],
)
    return Delague(d, γ1, γ2, coefficients, intercepts)
end

distribution(s::Delague) = s.dist

# NOTE: The Betina formulation doesn't use ambiguity sets?

end
