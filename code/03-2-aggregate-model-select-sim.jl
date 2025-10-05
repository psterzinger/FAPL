using JLD2, DataFrames, Printf, FiniteDiff, PrettyTables, LaTeXStrings

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul

datasets = filter(x -> occursin("MS", x), readdir(results_path))

results = [] 
models = [] 
penalties = [] 
options = [] 
scalings = [] 
rows = []
for (i, dataset) in enumerate(datasets)
    tmp = load(joinpath(results_path, dataset))["full_sim"]
    res = tmp.results
    if i == 1 
        results = res
        models = tmp.models 
        penalties = unique(res.pen)
        scalings = unique(res.scaling) 
        options = tmp.options 
    else 
        results = vcat(results, res) 
        models = (; models..., tmp.models...)
        penalties = Tuple(unique((penalties..., unique(res.pen)...)))
        scalings = Tuple(unique((scalings..., unique(res.scaling)...))) 
    end 
    push!(rows, nrow(res))
end 

sumry, hlines, subheaders = Main.FactorModelSimul.simul_summary(results, models; 
        excl_heywood = false, 
        excl_no_conv = false, 
        excl_grad_nz = false, 
        grad_cutoff = 1e-5,  
        heywood_cutoff = 1e-5, 
        print_only = false, 
        model_select = true
)

ns = unique(sumry.n[isa.(sumry.n, Int)])
mods = unique(sumry.Model[isa.(sumry.Model, Symbol)])
nconfigs = (length(unique(results.scaling)) - 1) * (length(unique(results.pen)) - 1) + 1 
reps = Int(nrow(filter(row -> row.model == mods[1] && row.n == ns[1], results)) / nconfigs)


sim_results = Main.FactorModelSimul.FMsimul(results, 
    (summary = sumry, hlines = hlines, subheaders = subheaders), 
    models, 
    penalties,
    scalings, 
    ns, 
    reps, 
    options, 
    true) 

@save joinpath(results_path, "03-2-full-sim-model-select.jld2") sim_results
