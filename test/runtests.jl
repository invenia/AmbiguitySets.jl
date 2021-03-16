using AmbiguitySets
using Distributions
using Random
using StatsBase: aweights
using StatsUtils: WeightedResampler
using Test

using AmbiguitySets:
    AmbiguitySet,
    BertsimasSet,
    BenTalSet,
    DelageSet,
    distribution


@testset "AmbiguitySets.jl" begin
    @testset "BertsimasSet" begin
        s = BertsimasSet(MvNormal(10, 0.25); Δ=fill(0.05, 10), Γ=1.0)

        @test length(s) == 10
        @test rand(MersenneTwister(123), s) == rand(MersenneTwister(123), distribution(s))
        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == BertsimasSet(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError BertsimasSet(MvNormal(10, 0.25); Δ=fill(-0.05, 10))
        @test_throws ArgumentError BertsimasSet(MvNormal(10, 0.25); Δ=fill(0.05, 9))
        @test_throws ArgumentError BertsimasSet(MvNormal(10, 0.25); Δ=fill(0.05, 10), Γ=-1.0)
        @test_logs(
            (:warn, "Budget should not exceed the distribution length"),
            BertsimasSet(MvNormal(10, 0.25); Δ=fill(0.05, 10), Γ=11.0)
        )
    end

    @testset "BenTalSet" begin
        s = BenTalSet(MvNormal(10, 0.25); Δ=0.025)

        @test length(s) == 10
        @test rand(MersenneTwister(123), s) == rand(MersenneTwister(123), distribution(s))
        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == BenTalSet(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError BenTalSet(MvNormal(10, 0.25); Δ=-0.5)
    end

    @testset "$S" for S in [DelageSet; YangSet]
        s = S(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
        )

        @test length(s) == 10
        @test rand(MersenneTwister(123), s) == rand(MersenneTwister(123), distribution(s))
        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == S(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError S(
            MvNormal(10, 0.25);
            γ1=-0.05,
            γ2=3.0,
        )

        @test_throws ArgumentError S(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=0.5,
        )

        if S === DelageSet
            @test_throws ArgumentError S(
                MvNormal(10, 0.25);
                γ1=0.05,
                γ2=3.0,
                coefficients=[1.0, 0.2],
                intercepts=[0.0],
            )
        end
        if S === YangSet
            d = MvNormal(10, 0.25)
            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                ξ̄=fill(0.05, 9),
                ξ̲=fill(-0.05, 10)
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                ξ̄=fill(0.05, 10),
                ξ̲=fill(-0.05, 9)
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                ξ̄=mean(d) .+ sqrt.(var(d)),
                ξ̲=mean(d) .+ sqrt.(var(d))
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                ξ̄=mean(d) .-  sqrt.(var(d)),
                ξ̲=mean(d) .-  sqrt.(var(d))
            )
        end
    end
end

@testset "DuSet" begin
    n = 10; n_obs = 100
    d = WeightedResampler(rand(n, n_obs), aweights(ones(n_obs)))
    s = DuSet(d; ϵ=0.01)

    @test length(s) == n
    @test isa(s, Sampleable)
    @test isa(s, AmbiguitySet)
    @test rand(MersenneTwister(123), s) == rand(MersenneTwister(123), distribution(s))
    @test s == DuSet(d)

    # Test constructor error cases
    @test_throws ArgumentError DuSet(d; ϵ=-0.01)
    @test_throws ArgumentError DuSet(d; Λ=-0.01)
    @test_throws ArgumentError DuSet(d; Q=rand(n+1,n))
end

@testset "AmbiguitySetEstimator.jl" begin
    n = 4
    M = 100000
    seed = 4444
    rng = MersenneTwister(seed)
    d = MvNormal(n, 0.25)
    data = Matrix(transpose(rand(rng, d, M)))
    @testset "Abstract Estimator: $S" for S in [BertsimasSet, BenTalSet, DelageSet]
        s = AmbiguitySets.estimate(AmbiguitySetEstimator{S}(), d, rand(M, n))
        @test isa(s, S)
    end
    @testset "Bertsimas Estimator" begin
        s = AmbiguitySets.estimate(
            BertsimasDataDrivenEstimator{BertsimasSet}(Δ_factor=7.7, Γ_factor=0.5), d, data
        )
        @test isa(s, BertsimasSet)
    end
    @testset "Delage Estimator" begin
        s = AmbiguitySets.estimate(DelageDataDrivenEstimator{DelageSet}(δ=0.025), d, data)
        @test isa(s, DelageSet)
    end
    @testset "Yang Estimator" begin
        s = AmbiguitySets.estimate(YangDataDrivenEstimator{YangSet}(Δ_factor=1.0), d, data)
        @test isa(s, YangSet)
    end
    @testset "Du Estimator" begin
        Λ_factor = 2.0
        s = AmbiguitySets.estimate(
            DuDataDrivenEstimator{DuSet}(Λ_factor=Λ_factor, norm_cone=Inf), d, data
        )
        @test isa(s, DuSet)
        @test s.Λ == Λ_factor * maximum(abs.(data))
    end
end
