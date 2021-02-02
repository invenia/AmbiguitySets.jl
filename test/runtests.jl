using AmbiguitySets
using Distributions
using Test

using AmbiguitySets:
    AmbiguitySet,
    UncertaintySet,
    Bertsimas


@testset "AmbiguitySets.jl" begin
    @testset "Bertsimas" begin
        # Really simple construction tests
        s = Bertsimas(MvNormal(4, 0.25), [0.1, 0.4, 0.2, 0.01], 3.0)

        @test isa(s, Sampleable)
        @test isa(s, AmbiguitySet)
        @test isa(s, UncertaintySet)

        s2 = Bertsimas(MvNormal(10, 0.25); uncertainty=fill(0.05, 10), budget=1.0)
        @test s2 == Bertsimas(MvNormal(10, 0.25))
    end
end
