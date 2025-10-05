using JLD2, DataFrames, CairoMakie, ColorSchemes, Statistics, LaTeXStrings, Optim 
using GeometryBasics: Point2f

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul


sim_results = load(joinpath(results_path, "03-2-full-sim-model-select.jld2"))["sim_results"]
sumry, hlines, subheader = Main.FactorModelSimul.simul_summary(sim_results.results, sim_results.models; 
                                                                excl_heywood = false, 
                                                                excl_no_conv = false, 
                                                                excl_grad_nz = false, 
                                                                grad_cutoff = 1e-4, 
                                                                heywood_cutoff = 1e-4, 
                                                                model_select = true, 
                                                                print_only = false)
sim_results = FactorModelSimul.FMsimul(sim_results.results, 
    (summary = sumry, hlines = hlines, subheaders = subheader),
    sim_results.models, sim_results.penalties, 
    sim_results.scalings,
    [50, 400, 1000],
    1000, 
    sim_results.options,
    true
) 

simul_reduction =  deepcopy(sim_results.summary.summary) 
Main.FactorModelSimul.forward_fill!(simul_reduction.n)
Main.FactorModelSimul.forward_fill!(simul_reduction.Model)
Main.FactorModelSimul.replace_empty_with_missing!(simul_reduction)
simul_reduction = dropmissing(simul_reduction) 
sort!(simul_reduction, [:Model, :n, :config])

df_wide = select(simul_reduction, [:Model, :n, Symbol("Correct AIC"), Symbol("Correct BIC"), :config])
df_long = stack(
  df_wide,
  [Symbol("Correct AIC"), Symbol("Correct BIC")],         
  variable_name = :MS,            
  value_name    = :perc_correct
)
df_long.MS .= replace.(df_long.MS,
                       "Correct AIC" => "AIC",
                       "Correct BIC" => "BIC"
)
df_long.config = convert.(Int64, map(x -> findfirst(y -> y == x, unique(df_long.config)[[1, 2, 4, 3, 5]]), df_long.config))

fig = Figure(size = (1000, 500))
bar_objs = BarPlot[]
n_values = unique(df_long.n)
configs = sort(unique(df_long.config))
Settings = unique(df_long.Model) 
MSs = unique(df_long.MS) 
ng = length(n_values)
nc = length(configs)
gw = 1.0
bw = gw / (nc + 1)
xs0 = [(i-1)*(gw + 0.5) for i in 1:ng]
centers = xs0 .+ gw / 2
xtl = [latexstring("\$ $n \$") for n in n_values]
colors = collect(cgrad(:viridis, length(configs), categorical  = true))
for (i_st, st) in enumerate(Settings), (j_ms, ms) in enumerate(MSs)
    row, col = i_st+1, j_ms
    if row == 2
        if col == 1 
            ax = Axis(
                fig[row, col],
                xticks = (centers, fill("", length(xtl))),
                yticks = (0:.25:1, [latexstring("\$ $(Int(100 * i)) \$") for i in 0:.25:1]),
                title = L"\mathrm{AIC}", 
                ylabelsize = 18, 
                ylabel = L"% \textrm{ samples}", 
                titlesize = 18, 
                )
                hidexdecorations!(ax, grid = false)
                hidespines!(ax, :b)
        else
            ax = Axis(
                fig[row, col],
                xticks = (centers, fill("", length(xtl))),
                yticks = (0:.25:1, [latexstring("\$ $(Int(100 * i)) \$") for i in 0:.25:1]),
                title = L"\mathrm{BIC}", 
                titlesize = 18
                )
            hidespines!(ax, :l, :b)
            hideydecorations!(ax, grid = false)
            hidexdecorations!(ax, grid = false)
        end 
        text!(
            ax,
            [Point2f(0, 1)];
            text  = [L"A_3"],
            align = (:left, :top),
            space = :relative,
            offset    = (4, -4),        
            fontsize  = 18
        )
    else
        if col == 1 
            ax = Axis(
                fig[row, col],
                xticks = (centers, xtl),
                yticks = (0:.25:1, [latexstring("\$ $(Int(100 * i)) \$") for i in 0:.25:1]),
                ylabelsize = 18, 
                ylabel = L"% \textrm{ samples}", 
                xlabel = L"n", 
                xlabelsize = 18
                )
        else 
              ax = Axis(
                fig[row, col],
                xticks = (centers, xtl),
                yticks = (0:.25:1, [latexstring("\$ $(Int(100 * i)) \$") for i in 0:.25:1]),
                xlabel = L"n", 
                xlabelsize = 18
                )
                hidespines!(ax, :l)
                hideydecorations!(ax, grid = false)
        end 
        text!(
            ax,
            [Point2f(0, 1)];
            text  = [L"B_3"],
            align = (:left, :top),
            space = :relative,
            offset    = (4, -4),        
            fontsize  = 18
        )

    end 
    hidespines!(ax, :t, :r)
    for (k, cfg) in enumerate(configs)
        xs = [ xs0[l] + k*bw for l in 1:ng ]
        ys = [ first(df_long[(df_long.Model .== st) .& (df_long.MS .== ms) .& 
                        (df_long.config .== cfg)  .& (df_long.n .== n), :perc_correct])
               for n in n_values ]
        b = barplot!(ax, xs, ys;
            width = bw,
            color = colors[k],
        )
        if i_st == 1 && j_ms == 1
            push!(bar_objs, b)
        end
    end
end

linkaxes!(filter(x -> x isa Axis, fig.content)...)

labels = [
    L"\textrm{None}",
    L"\textrm{Akaike}[n]",
    L"\textrm{Hirose}[n]",
    L"\textrm{Akaike}[n^{-1/2}]",
    L"\textrm{Hirose}[n^{-1/2}]",
]

Legend(
  fig[1, 1:2],
  bar_objs,
  labels,
  orientation = :horizontal,
  tellwidth = false,
  framevisible = false
)

fig

save(joinpath(figures_path, "model-select-perc-correct.pdf"), fig)
