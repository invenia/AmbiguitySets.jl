using AmbiguitySets
using Distributions
using Random
using Test

using AmbiguitySets:
    AmbiguitySet,
    BertsimasSet,
    BenTalSet,
    DelagueSet,
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

    @testset "DelagueSet" begin
        s = DelagueSet(
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
        @test s == DelagueSet(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError DelagueSet(
            MvNormal(10, 0.25);
            γ1=-0.05,
            γ2=3.0,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError DelagueSet(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=0.5,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError DelagueSet(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
            coefficients=[1.0, 0.2],
            intercepts=[0.0],
        )
    end
end
