using AmbiguitySets
using Documenter

makedocs(;
    modules=[AmbiguitySets],
    authors="Invenia Technical Computing Corporation",
    repo="https://gitlab.invenia.ca/invenia/research/AmbiguitySets.jl/blob/{commit}{path}#L{line}",
    sitename="AmbiguitySets.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
    strict=true,
    checkdocs=:exports,
)
