"""
    AmbiguitySetEstimator{S<:AmbiguitySet}

Abstract type for parameter estimation algorithms.
"""
abstract type AbstractAmbiguitySetEstimator{S<:AmbiguitySet} end

struct AmbiguitySetEstimator{S<:AmbiguitySet} <: AbstractAmbiguitySetEstimator{S} end

"""
    estimate(::Type{<:AmbiguitySetEstimator{S}}, d, data; kwargs...) where {S<:AmbiguitySet}

Constructs an `AmbiguitySet` by estimating appropriate parameters from the predictive distribution and raw samples.

Attributes:
 - `d`: Predictive distribution.
 - `data`: Raw samples.
"""
estimate(::AbstractAmbiguitySetEstimator{S}, d, data::Array{Float64,2}; kwargs...) where {S<:AmbiguitySet} = S(d; kwargs...)

"""
    DelageDataDrivenEstimator{S, T} <: AmbiguitySetEstimator{S}

Based on the Depage's paper section 3.4: https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems
"""
struct DelageDataDrivenEstimator{S, T} <: AbstractAmbiguitySetEstimator{S}
    δ::T
    function DelageDataDrivenEstimator{S}(;
        δ::T=0.001
    ) where {S<:DelageSet, T<:Real}
        (δ >= 0.0 && δ <= 1.0) || throw(ArgumentError("δ must be: 0 <= δ <= 1"))
        return new{S, T}(δ)
    end
end

"""
    estimate(::Type{<:AmbiguitySetEstimator{S}}, d, data; kwargs...) where {S<:AmbiguitySet}

Constructs an `DelageSet` by estimating appropriate parameters from the predictive distribution and raw samples.

Attributes:
 - `d`: Predictive distribution.
 - `ξ`: Raw samples.
"""
function estimate(estimator::DelageDataDrivenEstimator{S, T}, d, ξ::Array{Float64,2}; kwargs...)::S where {S<:DelageSet, T<:Real} 
    δ = estimator.δ
    means = Distributions.mean(d)
    m = length(means)
    inv_sqrt_cov = inv(Matrix(sqrt(cov(d))))
    M = size(ξ,1)
    sqrt_M = sqrt(M)

    δ̄ = 1 - sqrt(1-δ)

    R̂ = maximum([norm(inv_sqrt_cov * (ξ[i,:] - means), 2) for i = 1:M])

    R̄ = (1 - (R̂^2 + 2) * ((2 + sqrt(2 * log(4 / δ̄))) / (sqrt_M)))^(-1/2)  * R̂

    ᾱ = (R̄^2 / sqrt_M) * (sqrt(1 - (m / R̄^4)) + sqrt(log(4 / δ̄)))

    β̄ = (R̄^2 / M) * (2 + sqrt(2 * log(2 / δ̄)))^2

    γ1 = β̄ / (1 - ᾱ - β̄)

    γ2 = (1 + β̄) / (1 - ᾱ - β̄)

    return S(d; γ1=γ1, γ2=γ2, kwargs...)
end
