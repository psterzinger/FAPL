using JLD2, DataFrames, Optim, Latexify 

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul


sim_results = load(joinpath(results_path, "03-2-full-sim-model-select.jld2"))["sim_results"]
selected_model = :B_3 

selected_results = filter(row -> row.model == selected_model, sim_results.results)
selected_results[:,"config"] .= "[" .* string.(selected_results.pen) .* "," .* string.(selected_results.scaling) .* "]"
configs = unique(selected_results.config)[[5, 1, 3, 2, 4]]
num_configs = length(configs)
selected_qs = map(x -> x.q[2], selected_results.FMfit)
nan_inds = isnan.(selected_qs)
selected_results = selected_results[.!nan_inds,:]
unique_qs = sort(unique(selected_qs[.!nan_inds]))
ns = repeat(unique(selected_results.n), inner = length(unique_qs))
qs = repeat(unique_qs, outer = length(unique(ns)))
BIC_table = DataFrame(n = ns, q = qs)
for config in configs
    BIC_table[!, config] .= NaN 
end 

for (i, n) in enumerate(unique(ns))  
    results_n = filter(x -> x.n == n, selected_results) 
    for (j, config) in enumerate(configs) 
        results_c = filter(x -> x.config == config, results_n) 
        n_runs = nrow(results_c)
        q_perc = fill(NaN, length(unique_qs))
        for (k, q) in enumerate(unique_qs) 
            q_perc[k] = sum(map(x -> x.q[2], results_c.FMfit) .== q) / n_runs 
        end 
        BIC_table[(i - 1) * length(unique_qs) + 1 : i * length(unique_qs), j + 2] .= q_perc 
    end 
end 

for i in axes(BIC_table, 1) 
    for j in axes(BIC_table, 2) 
        c = BIC_table[i, j]
        if !isa(c, Int)
            BIC_table[i, j] = round.(100 * c, digits=1)
        end 
    end 
end 

BIC_table
latexify(BIC_table; env = :table, booktabs = true, latex = false) |> print
