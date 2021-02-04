using AmbiguitySets
using Distributions
using Test

using AmbiguitySets:
    AmbiguitySet,
    Bertsimas,
    BenTal,
    Delague,
    distribution


@testset "AmbiguitySets.jl" begin
    # Mostly just some simple constructor tests
    @testset "Bertsimas" begin
        s = Bertsimas(MvNormal(10, 0.25); Δ=fill(0.05, 10), Γ=1.0)

        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == Bertsimas(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError Bertsimas(MvNormal(10, 0.25); Δ=fill(-0.05, 10))
        @test_throws ArgumentError Bertsimas(MvNormal(10, 0.25); Δ=fill(0.05, 9))
    end

    @testset "BenTal" begin
        s = BenTal(MvNormal(10, 0.25); Δ=0.05)

        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == BenTal(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError BenTal(MvNormal(10, 0.25); Δ=-0.5)
    end

    @testset "Delague" begin
        # Really simple construction tests
        s = Delague(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test distribution(s) == MvNormal(10, 0.25)
        @test s == Delague(MvNormal(10, 0.25))

        # Test constructor error cases
        @test_throws ArgumentError Delague(
            MvNormal(10, 0.25);
            γ1=-0.05,
            γ2=3.0,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError Delague(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=0.5,
            coefficients=[1.0],
            intercepts=[0.0],
        )

        @test_throws ArgumentError Delague(
            MvNormal(10, 0.25);
            γ1=0.05,
            γ2=3.0,
            coefficients=[1.0, 0.2],
            intercepts=[0.0],
        )
    end
end
