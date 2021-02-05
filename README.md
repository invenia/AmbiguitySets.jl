# AmbiguitySets.jl

[![Docs: Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://docs.invenia.ca/invenia/research/AmbiguitySets.jl)

A minimal package of ambiguity set type definitions for use in
[AmbiguitySetForecasters.jl](https://gitlab.invenia.ca/invenia/research/AmbiguitySetForecasters.jl) and
[PortfolioOptimization.jl](https://gitlab.invenia.ca/invenia/PortfolioOptimization.jl).

## Implemented Types

- `BertsimasSet`
- `BenTalSet`
- `DelagueSet`

## FAQ

__Why is `AmbiguitySet <: Sampleable`?__

1. If you think of ambiguity sets as representing a bounded infinite set of distributions then we can think of them as compound distributions, similar to mixture distributions (finite set of distributions).
2. Reuses existing APIs in Forecasters.jl, IndexedDistributions.jl and PortfolioOptimization.jl, which often assume the input is an `IndexedSampleable`.

__Why don't these types live with the formulation code in PortfolioOptimization.jl?__

As discussed above, these types provide a convenient way of interacting with the rest of our forecaster / decision engine pipeline.

1. AmbiguitySetForecasters.jl can construct these sets using raw data and knowledge of a predictive distribution without caring about how the formulation will use the parameters.
2. PortfolioOptimization.jl can take these sets without caring whether they came from AmbiguitySetForecasters.jl
3. We avoid storing/passing raw data between the forecaster and formulation which can be expensive for large fetched datasets.
