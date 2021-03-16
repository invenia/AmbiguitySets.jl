module AmbiguitySets

using Distributions
using LinearAlgebra
using Random

export AmbiguitySet, AmbiguitySetEstimator, BertsimasSet, BenTalSet, 
    DuSet, DelageSet, YangSet
export BertsimasDataDrivenEstimator, DuDataDrivenEstimator, DelageDataDrivenEstimator, 
    estimate, YangDataDrivenEstimator

const ContinuousMultivariateSampleable = Sampleable{Multivariate, Continuous}

"""
    AmbiguitySet <: Sampleable

Defines the ambiguity related to a contained distribution, often used in
Robust Optimization (RO) and Distributionally Robust Optimization (DRO) problems.
It represents a bounded infinite set of distributions.

For more information on how `AmbiguitySet`s are used for RO and DRO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
abstract type AmbiguitySet{T<:Real, D<:ContinuousMultivariateSampleable} <: ContinuousMultivariateSampleable end

Base.length(s::AmbiguitySet) = length(distribution(s))

function Distributions._rand!(
    rng::AbstractRNG, s::AmbiguitySet{T}, x::AbstractVector{T}
) where {T<:Real}
    return rand!(rng, distribution(s), x)
end

# NOTE: Technically we could probably implement `rand` for `Bertsimas` and `BenTal` by
# uniformly augmenting the samples from the underlying MvNormal with the uncertainty
# estimate. I don't think anything like that would be possible for Delage though.

"""
    BertsimasSet <: AmbiguitySet

```math
\\left\\{ \\mu \\; \\middle| \\begin{array}{ll}
s.t.  \\quad \\mu_i \\leq \\hat{r}_i + z_i \\Delta_i \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad \\mu_i \\geq \\hat{r}_i - z_i \\Delta_i  \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad z_i \\geq 0 \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad z_i \\leq 1 \\quad \\forall i = 1:\\mathcal{N} \\\\
\\quad \\quad \\sum_{i}^{\\mathcal{N}} z_i \\leq \\Gamma \\\\
\\end{array}
\\right\\} \\\\
```

Attributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uncertainty around mean. (default: std(d) / 5)
- `Γ::Float64`: Number of assets in worst case. (default: 0.1 * length(d))

References:
- Bertsimas, D. e Sim, M. (2004). The price of robustness. Operations research, 52(1):35–53.

For more information on how Bertsimas' uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct BertsimasSet{T<:Real, D<:ContinuousMultivariateSampleable} <: AmbiguitySet{T, D}
    d::D
    Δ::Vector{T}
    Γ::T

    # Inner constructor for validating arguments
    function BertsimasSet{T, D}(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == length(Δ) || throw(ArgumentError(
            "Distribution ($(length(d))) and Δ ($(length(Δ))) are not the same length"
        ))
        all(>=(0), Δ) || throw(ArgumentError("All uncertainty deltas must be >= 0"))
        Γ >= 0 || throw(ArgumentError("Budget must be >= 0"))
        Γ <= length(d) || @warn "Budget should not exceed the distribution length"
        return new{T, D}(d, Δ, Γ)
    end
end

# Default outer constructor
BertsimasSet(d::D, Δ::Vector{T}, Γ::T) where {T<:Real, D<:ContinuousMultivariateSampleable} = BertsimasSet{T, D}(d, Δ, Γ)

# Kwarg constructor with defaults
function BertsimasSet(
    d::AbstractMvNormal;
    Δ=default_bertsimas_delta(d),
    Γ=default_bertsimas_budget(d),
)
    return BertsimasSet(d, Δ, Γ)
end

# Not sure about these defaults, but it seems like something we should support?
default_bertsimas_delta(d::AbstractMvNormal) = sqrt.(var(d)) ./ 5
default_bertsimas_budget(d::Sampleable) = 0.1 * length(d)

distribution(s::BertsimasSet) = s.d

"""
    BenTalSet <: AmbiguitySet

```math
\\left\\{ \\mu \\; \\middle| \\begin{array}{ll}
s.t.  \\quad \\sqrt{(\\hat{r} - \\mu) ' \\Sigma^{-1} (\\hat{r} - \\mu)} \\leq \\delta \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `Δ::Array{Float64,1}`: Uniform uncertainty around mean. (default: 0.025)

References:
- Ben-Tal, A. e Nemirovski, A. (2000). Robust solutions of linear programming problems contaminated with uncertain data. Mathematical programming, 88(3):411–424.

For more information on how BenTal uncertainty sets are used for RO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct BenTalSet{T<:Real, D<:ContinuousMultivariateSampleable} <: AmbiguitySet{T, D}
    d::D
    Δ::T

    # Inner constructor for validating arguments
    function BenTalSet{T, D}(d::D, Δ::T) where {T<:Real, D<:ContinuousMultivariateSampleable}
        Δ >= 0 || throw(ArgumentError("Uncertainty delta must be >= 0"))
        return new{T, D}(d, Δ)
    end
end

# Default outer constructor
BenTalSet(d::D, Δ::T) where {T<:Real, D<:ContinuousMultivariateSampleable} = BenTalSet{T, D}(d, Δ)

# Kwarg constructor with default delta value
BenTalSet(d::AbstractMvNormal; Δ=0.025) = BenTalSet(d, Δ)

distribution(s::BenTalSet) = s.d

"""
    DelageSet <: AmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad (\\mathbb{E} [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E} [r] - \\hat{r}) \\leq \\gamma_1 \\\\
\\quad \\quad \\mathbb{E} [ (r - \\hat{r}) ' (r - \\hat{r}) ] \\leq \\gamma_2 \\Sigma \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0). (default: std(dist) / 5)
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1). (default: 3.0)
- `coefficients::Vector{Float64}`: Piece-wise utility coeficients (default [1.0]).
- `intercepts::Vector{Float64}`: Piece-wise utility intercepts (default [0.0]).

References:
- Delage paper on moment uncertainty (what I implemented): https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems

For more information on how Delage ambiguity sets are used for DRO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct DelageSet{T<:Real, D<:ContinuousMultivariateSampleable} <: AmbiguitySet{T, D}
    d::D
    γ1::T
    γ2::T
    coefficients::Vector{T}
    intercepts::Vector{T}

    # Inner constructor for validating arguments
    function DelageSet{T, D}(
        d::D, γ1::T, γ2::T, coefficients::Vector{T}, intercepts::Vector{T}
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
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
function DelageSet(
    d::D, γ1::T, γ2::T, coefficients::Vector{T}, intercepts::Vector{T}
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    DelageSet{T, D}(d, γ1, γ2, coefficients, intercepts)
end

# Kwarg constructor with defaults
function DelageSet(
    d::AbstractMvNormal;
    γ1=default_delage_γ1(d), γ2=3.0, coefficients=[1.0], intercepts=[0.0],
)
    return DelageSet(d, γ1, γ2, coefficients, intercepts)
end

default_delage_γ1(d::AbstractMvNormal) = first(sqrt.(var(d)) ./ 5)

distribution(s::DelageSet) = s.d


"""
    YangSet <: AmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad (\\mathbb{E} [r] - \\hat{r}) ' \\Sigma^{-1} (\\mathbb{E} [r] - \\hat{r}) \\leq \\gamma_1 \\\\
\\quad \\quad \\mathbb{E} [ (r - \\hat{r}) ' (r - \\hat{r}) ] \\leq \\gamma_2 \\Sigma \\\\
\\quad \\quad \\underline{\\xi} \\leq \\xi \\leq \\bar{\\xi} \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::Sampleable{Multivariate, Continous}`: The parent distribution with an uncertain mean
- `γ1::Float64`: Uniform uncertainty around the mean (has to be greater than 0). (default: std(dist) / 5)
- `γ2::Float64`: Uncertainty around the covariance (has to be greater than 1). (default: 3.0)
- `coefficients::Vector{Float64}`: Piece-wise utility coeficients (default [1.0]).
- `intercepts::Vector{Float64}`: Piece-wise utility intercepts (default [0.0]).
- `ξ̄::Vector{T}`: Suport upper limits
- `ξ̲::Vector{T}`: Suport lower limits

References:
- Li Yang paper on moment uncertainty and CVAR: https://www.hindawi.com/journals/jam/2014/784715/

For more information on how Yang ambiguity sets are used for DRO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct YangSet{T<:Real, D<:ContinuousMultivariateSampleable} <: AmbiguitySet{T, D}
    d::D
    γ1::T
    γ2::T
    ξ̲::Vector{T}
    ξ̄::Vector{T}

    # Inner constructor for validating arguments
    function YangSet{T, D}(
        d::D, γ1::T, γ2::T, ξ̲::Vector{T}, ξ̄::Vector{T}
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == length(ξ̄) || throw(ArgumentError(
            "Distribution ($(length(d))) and ξ̄ ($(length(ξ̄))) are not the same length"
        ))
        length(d) == length(ξ̲) || throw(ArgumentError(
            "Distribution ($(length(d))) and ξ̄ ($(length(ξ̲))) are not the same length"
        ))
        means = mean(d)
        all(ξ̄ .>= means) || throw(ArgumentError("ξ̄ must be >= mean(d)"))
        all(ξ̲ .<= means) || throw(ArgumentError("ξ̲ must be <= mean(d)"))
        γ1 >= 0 || throw(ArgumentError("γ1 must be >= 0"))
        γ2 >= 1 || throw(ArgumentError("γ2 must be >= 1"))
        return new{T, D}(d, γ1, γ2, ξ̲, ξ̄)
    end
end

# Default outer constructor
function YangSet(
    d::D, γ1::T, γ2::T, ξ̲::Vector{T}, ξ̄::Vector{T}
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    YangSet{T, D}(d, γ1, γ2, ξ̲, ξ̄)
end

# Kwarg constructor with defaults
function YangSet(
    d::AbstractMvNormal;
    γ1=default_delage_γ1(d), γ2=3.0, 
    ξ̲=(mean(d) .- default_bertsimas_delta(d)), ξ̄=(mean(d) .+ default_bertsimas_delta(d))
)
    return YangSet(d, γ1, γ2, ξ̲, ξ̄)
end

distribution(s::YangSet) = s.d

"""
    DuSet <: AmbiguitySet

```math
\\left\\{ r  \\; \\middle| \\begin{array}{ll}
s.t.  \\quad d_w(P, \\hat{P}_N) \\leq \\epsilon \\\\
\\quad \\quad || \\xi || \\leq \\Lambda \\\\
\\quad \\quad [\\xi; 0] + [0_{m x 1}; \\Lambda] \\in K \\\\
\\quad \\quad K = {[\\omega, \\pi] \\in R^m x R: \\pi \\geq ||\\omega||^*} \\\\
\\end{array}
\\right\\} \\\\
```

Atributes:
- `d::ContinuousMultivariateSampleable`: Samples from the parent distribution
- `ϵ::Float64`: Wasserstein distance from sampled distribution (has to be greater than 0). (default: 0.01)
- `Λ::Float64`: Uncertainty around sampled values (has to be greater than 0). (default: maximum(std(d)))

References:
- NingNing paper on Wasserstein DRO (Corollary 1-3): https://ieeexplore.ieee.org/abstract/document/9311154

For more information on how NingNing ambiguity sets are used for DRO, please review
the PortfolioOptimization.jl [docs](https://invenia.pages.invenia.ca/PortfolioOptimization.jl/).
"""
struct DuSet{T<:Real, D<:ContinuousMultivariateSampleable} <: AmbiguitySet{T, D}
    d::D
    ϵ::T
    Λ::T
    Q::Array{T,2}
    norm_cone::Real

    # Inner constructor for validating arguments
    function DuSet{T, D}(
        d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
    ) where {T<:Real, D<:ContinuousMultivariateSampleable}
        length(d) == size(Q,1) == size(Q,2) || throw(ArgumentError(
            "Distribution ($(length(d))) and Q ($(size(Q,2))) must have coherent dimensions (m and mxm)"
        ))
        ϵ >= 0 || throw(ArgumentError("ϵ must be >= 0"))
        Λ >= 0 || throw(ArgumentError("Λ must be >= 0"))
        return new{T, D}(d, ϵ, Λ, Q, norm_cone)
    end
end

# Default outer constructor
function DuSet(
    d::D, ϵ::T, Λ::T, Q::Array{T,2}, norm_cone::Real
) where {T<:Real, D<:ContinuousMultivariateSampleable}
    DuSet{T, D}(d, ϵ, Λ, Q, norm_cone)
end

# Kwarg constructor with defaults
function DuSet(
    d::ContinuousMultivariateSampleable;
    ϵ=0.01,
    norm_cone=Inf,
    Λ=default_DuSet_lambda(d, norm_cone),
    Q=diagm(ones(length(d)))
)
    return DuSet(d, ϵ, Λ, Q, norm_cone)
end

distribution(s::DuSet) = s.d

default_DuSet_lambda(d::Sampleable, norm_cone::Real; num_samples::Int=20, rng::AbstractRNG=MersenneTwister(123)) = maximum(
    norm.(eachrow(rand(rng, d, num_samples)), norm_cone)
)

include("AmbiguitySetEstimator.jl")

end
