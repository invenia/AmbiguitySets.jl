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
abstract type AmbiguitySet{F<:Multivariate, S<:Continuous, D<:Sampleable{F, S}} <: Sampleable{F, S} end

Base.rand(s::AmbiguitySet) = rand(distribution(s))

"""
    UncertaintySet <: AmbiguitySet

The supertype of all Uncertainty Sets. This is the class of purelly robust Ambiguity Sets.
The uncertainty set bounds the values which the random vector can assume.

NOTE: I'm not entire clear if this is a necessary subtype
"""
abstract type UncertaintySet{F<:Multivariate, S<:Continuous, D<:Sampleable{F, S}} <: AmbiguitySet{F, S, D} end

"""
    Bertsimas <: UncertaintySet

Atributes:
- `dist::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `uncertainty::Array{Float64,1}` (latex notation ``\\Delta``): Uncertainty around mean. (default: std(dist) / 5)
- `budget::Float64` (latex notation ``\\Gamma``): Number of assets in worst case. (default: 0.1 * length(dist))

For more information on how Bertsimas' uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct Bertsimas{F<:Multivariate, S<:Continuous, D<:Sampleable{F, S}} <: UncertaintySet{F, S, D}
    dist::D
    uncertainty::Vector{Float64}
    budget::Float64
end

# Not sure about these defaults, but it seems like something we should support?
default_bertsimas_uncertainty(d::AbstractMvNormal) = sqrt.(diag(cov(d))) ./ 5
default_bertsimas_budget(d::Sampleable{Multivariate}) = 0.1 * length(d)

function Bertsimas(
    d::AbstractMvNormal;
    uncertainty=default_bertsimas_uncertainty(d),
    budget=default_bertsimas_budget(d)
)
    return Bertsimas(d, uncertainty, budget)
end

distribution(s::Bertsimas) = s.dist



end
