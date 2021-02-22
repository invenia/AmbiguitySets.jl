using AmbiguitySets
using Distributions
using Random
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

    @testset "$S" for S in [DelageSet; LiYangSet]
        s = S(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
            coefficients=[1.0],
            intercepts=[0.0],
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
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError S(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=0.5,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError S(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
            coefficients=[1.0, 0.2],
            intercepts=[0.0],
        )

        if S === LiYangSet
            d = MvNormal(10, 0.25)
            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                coefficients=[1.0],
                intercepts=[0.0],
                ξ̄=fill(0.05, 9),
                ξ̲=fill(-0.05, 10)
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                coefficients=[1.0],
                intercepts=[0.0],
                ξ̄=fill(0.05, 10),
                ξ̲=fill(-0.05, 9)
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                coefficients=[1.0],
                intercepts=[0.0],
                ξ̄=mean(d) .+ sqrt.(var(d)),
                ξ̲=mean(d) .+ sqrt.(var(d))
            )

            @test_throws ArgumentError S(
                d;
                γ1=0.05,
                γ2=3.0,
                coefficients=[1.0],
                intercepts=[0.0],
                ξ̄=mean(d) .-  sqrt.(var(d)),
                ξ̲=mean(d) .-  sqrt.(var(d))
            )
        end
    end
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
    @testset "Delague Estimator" begin
        s = AmbiguitySets.estimate(DelageDataDrivenEstimator{DelageSet}(δ=0.025), d, data)
        @test isa(s, DelageSet)
    end
end
