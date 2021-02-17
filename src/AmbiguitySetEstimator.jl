abstract type AmbiguitySetEstimator{S<:AmbiguitySet} end

estimate(estimator::AmbiguitySetEstimator{S}, d, data; kwargs...)::S where {S<:AmbiguitySet} = S(d; kwargs...)

"""
Based on the Depage's paper section 3.4: https://www.researchgate.net/publication/220244490_Distributionally_Robust_Optimization_Under_Moment_Uncertainty_with_Application_to_Data-Driven_Problems
"""
struct DelageDataDrivenEstimator{S} <: AmbiguitySetEstimator{S} where {S::DelageSet, T<:Real}
    δ::T
    function DelageDataDrivenEstimator{S,T}(;
        δ::T=0.001
    ) where {S<:DelageSet, T<:Real}
        (δ >= 0.0 && δ <= 1.0) || throw(ArgumentError("δ must be: 0 <= δ <= 1"))
        return new{S, T}(δ)
    end
end

function estimate(estimator::DelageDataDrivenEstimator{S}, d, ξ; kwargs...)::T where {S<:DelageSet} 
    δ = estimator.δ
    means = Distributions.mean(d)
    inv_sqrt_cov = inv(Matrix(sqrt(cov(d))))
    M = size(ξ,1)
    sqrt_M = sqrt(M)

    δ̄ = 1 - sqrt(1-δ)

    R̂ = maximum([norm(d.Σ.chol.L \ (ξ - means), 2) for ξ in fdata])

    R̄ = (1 - (R̂^2 + 2) * ((2 + sqrt(2 * log(4 / δ̄))) / (sqrt_M)))^(-1/2)  * R̂

    ᾱ = (R̄^2 / sqrt_M) * (sqrt(1 - (M / R̄^4)) + sqrt(log(4 / δ̄)))

    β̄ = (R̄^2 / M) * (2 + sqrt(2 * log(2 / δ̄)))^2

    γ1 = β̄ / (1 - ᾱ - β̄)

    γ2 = (1 + β̄) / (1 - ᾱ - β̄)

    return S(distribution(dist); γ1=γ1, γ2=γ2, kwargs...)
end
