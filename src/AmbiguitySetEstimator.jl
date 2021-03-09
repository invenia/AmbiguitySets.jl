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
    BertsimasDataDrivenEstimator{S, T} <: AmbiguitySetEstimator{S}

Parameter estimator for Bertsimas method
"""
struct BertsimasDataDrivenEstimator{S, T} <: AbstractAmbiguitySetEstimator{S}
    Δ_factor::T
    Γ_factor::T
    function BertsimasDataDrivenEstimator{S}(;
        Δ_factor::T=0.2,
        Γ_factor::T=0.1
    ) where {S<:BertsimasSet, T<:Real}
        0.0 <= Δ_factor || throw(ArgumentError("Δ_factor must be: 0 <= Δ_factor"))
        0.0 <= Γ_factor <= 1.0 || throw(ArgumentError("Δ_factor must be: 0 <= Γ_factor <= 1"))
        return new{S, T}(Δ_factor, Γ_factor)
    end
end

"""
    estimate(::Type{<:BertsimasDataDrivenEstimator{S}}, d, data; kwargs...) where {S<:BertsimasSet, T<:Real}

Constructs an `BertsimasSet` by estimating appropriate parameters from the predictive distribution.

Attributes:
 - `d`: Predictive distribution.
 - `data`: Raw samples. Not used.
"""

function estimate(estimator::BertsimasDataDrivenEstimator{S, T}, d, data::Array{Float64,2}; kwargs...)::S where {S<:BertsimasSet, T<:Real}
    return S(d; Δ=estimator.Δ_factor*sqrt.(var(d)), Γ=estimator.Γ_factor*length(d), kwargs...)
end

"""
    DelageDataDrivenEstimator{S, T} <: AmbiguitySetEstimator{S}

Based on the Delage's paper section 3.4: https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems
"""
struct DelageDataDrivenEstimator{S, T} <: AbstractAmbiguitySetEstimator{S}
    δ::T
    function DelageDataDrivenEstimator{S}(;
        δ::T=0.001
    ) where {S<:DelageSet, T<:Real}
        0.0 <= δ <= 1.0 || throw(ArgumentError("δ must be: 0 <= δ <= 1"))
        return new{S, T}(δ)
    end
end

"""
    estimate(::Type{<:DelageDataDrivenEstimator{S, T}}, d, ξ; kwargs...) where {S<:DelageSet, T<:Real}

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

"""
    BoxDuDataDrivenEstimator{S, T} <: AmbiguitySetEstimator{S}

Parameter estimator for Bertsimas method
"""
struct BoxDuDataDrivenEstimator{S, T} <: AbstractAmbiguitySetEstimator{S}
    Λ_factor::T
    function BoxDuDataDrivenEstimator{S}(;
        Λ_factor::T=1.0
    ) where {S<:BoxDuSet, T<:Real}
        1.0 <= Λ_factor || throw(ArgumentError("Λ_factor must be: 1 <= Δ_factor"))
        return new{S, T}(Λ_factor)
    end
end

"""
    estimate(::Type{<:BoxDuDataDrivenEstimator{S}}, d, data; kwargs...) where {S<:BoxDuSet, T<:Real}

Constructs an `BoxDuSet` by estimating appropriate parameters from the predictive distribution.

Attributes:
 - `d`: Predictive distribution.
 - `data`: Raw samples. Not used.
"""

function estimate(estimator::BoxDuDataDrivenEstimator{S, T}, d, data::Array{Float64,2}; kwargs...)::S where {S<:BoxDuSet, T<:Real}
    return S(d; Λ=estimator.Λ_factor*maximum(data), kwargs...)
end
