using JLD2, DataFrames, CairoMakie, ColorSchemes, LaTeXStrings

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul

function to_subscript(s::AbstractString)
    base_subscript_code = 0x2080
    result = Char[]
    i = 1
    while i <= length(s)
        c = s[i]
        if c == '_' && i < length(s) && isdigit(s[i + 1])
            i += 1
            while i <= length(s) && isdigit(s[i])
                digit = parse(Int, s[i])  
                push!(result, Char(base_subscript_code + digit))
                i += 1
            end
            continue
        else
            push!(result, c)
        end
        i += 1
    end
    
    return String(result)
end

function strip_named_fields(s::String)
    return replace(s, r"\b\w+\s*=\s*" => "")
end

sim_results = load(joinpath(results_path, "01-2-full-sim.jld2"))["sim_results"]
simul_reduction = Main.FactorModelSimul.reduce_simul(sim_results, (), (); 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    latex = false, 
    grad_cutoff = 1e-4, 
    heywood_cutoff = 1e-4).data
sort!(simul_reduction, [:Model, :n])
permute_inds = fill(0, size(simul_reduction, 1))
for i in 1:Int(length(permute_inds) / 5)
    permute_inds[5 * (i - 1) + 1 : 5 * i] .= (i - 1) * 5 .+ [5, 1, 2, 3, 4]
end 
simul_reduction = simul_reduction[permute_inds,:]
configs = unique(simul_reduction.config) 

col = collect(cgrad(:viridis, length(configs), categorical  = true))
figsize = (1000, 500)
fig = Figure(size = figsize)
for (i, n) in enumerate(unique(simul_reduction.n))
    perc_tab_long = filter(:n => x -> x == n, simul_reduction)[!,["Model","% used", "config"]]
    ordered_models = sort(unique(perc_tab_long.Model))
    col = collect(cgrad(:viridis, length(configs), categorical  = true))
    config_names = 1:length(configs) 
    ax = Axis(fig[2,i], 
                xticks = (1:length(ordered_models), latexstring.(ordered_models)), 
                subtitle = latexstring("n = $n"),
                yticks = (0:.25:1, [latexstring("\$ $(Int(100 * i)) \$") for i in 0:.25:1]),
                ylabelsize = 18, 
                ylabel = L"% \textrm{ Heywood cases}", 
                xticklabelsize = 18)
    mod_convert = map(x -> findfirst(y -> y == x, sort(unique(perc_tab_long.Model))), perc_tab_long.Model)
    config_convert = map(x -> findfirst(y -> y == x, configs), perc_tab_long.config)
    barplot!(ax, mod_convert, 1 .- Vector{Float64}(perc_tab_long[!,"% used"]),
            dodge = config_convert,
            color = col[config_convert])

    if i > 1 
        hidespines!(ax, :t, :r, :l)
        hideydecorations!(ax, grid = false)
    else 
        hidespines!(ax, :t, :r)
    end 
end 

labels = [
    L"\textrm{None}",
    L"\textrm{Akaike}[n]",
    L"\textrm{Hirose}[n]", 
    L"\textrm{Akaike}[n^{-1/2}]",
    L"\textrm{Hirose}[n^{-1/2}]"
]

elements = [PolyElement(polycolor = col[i]) for i in 1:length(labels)]
title = L"\textbf{\textrm{Penalty}}"
Legend(fig[1,1:3], 
    elements, 
    labels, 
    orientation = :horizontal,
    tellwidth = false,
    framevisible = false)
Label(
    fig[3, 1:3],
    L"\textrm{Setting}", 
    fontsize = 20,
    padding = (0, 0, 0, 0),
)

display(fig) 
save(joinpath(figures_path, "perc-HW.pdf"), fig)
