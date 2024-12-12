using Pkg
Pkg.add("Documenter")
using Documenter
Pkg.activate(".")
Pkg.instantiate()
makedocs(sitename = "ARNO AND SEBASTIAN DYCOPS 2024", pages = ["Optimal Data Gathering for Missing Physics.md"])
deploydocs(; repo = "github.com/arno-papers/DYCOPS2024", devbranch = "main", push_preview = true)
