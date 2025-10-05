using JLD2, DataFrames, CairoMakie, ColorSchemes, Statistics, LaTeXStrings, Optim 
using GeometryBasics: Point2f

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul

function compute_MSE(thetas, theta_true)
    compute_bias(thetas, theta_true).^2 .+ compute_var(thetas, theta_true)
end

function compute_bias(thetas, theta_true)
    if isempty(thetas) 
        return NaN 
    end
    Statistics.mean(map(x -> x .- theta_true, thetas))
end

function compute_var(thetas, theta_true)
    if isempty(thetas) 
        return NaN 
    end
    Statistics.mean(map(x -> ((x .- theta_true).^2), thetas))
end

function compute_RMSE(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    # mean(map(x -> sqrt(mean((x - theta_true).^2)), thetas))
    sqrt.(compute_MSE(thetas, theta_true)) 
end

function compute_abs_bias(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    Statistics.mean(map(x -> abs.(x - theta_true), thetas))
end 

function compute_PU(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    inds = map(x -> !any(isnan.(x)), thetas) 
    Statistics.mean(map(x -> (x .- theta_true) .< 0.0 , thetas[inds]))
end

function apply_fontscale!(ax::Axis, f::Float64)
    ax.titlegap[] = 10 * f
    ax.titlesize[] = 18 * f
    ax.xlabelsize[] = 18 * f
    ax.ylabelsize[] = 18 * f
    ax.xticklabelsize[] = 16 * f
    ax.yticklabelsize[] = 16 * f
end

second_element(x) = isa(x, Tuple) ? x[2] : x

summary_funs = (Bias = compute_bias, 
    Abs_Bias = compute_abs_bias,
    RMSE = compute_RMSE, 
    PU = compute_PU
)

sim_results = load(joinpath(results_path, "01-2-full-sim.jld2"))["sim_results"]

sumr, hline = Main.FactorModelSimul.simul_summary(sim_results.results, sim_results.models,
   summary_funs; 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-4, 
    heywood_cutoff = 1e-4, 
    model_select = false, 
    print_only = false
)

sim_results.summary.summary .= sumr 
sim_results.summary.hlines .= hline

simul_reduction = Main.FactorModelSimul.reduce_simul(sim_results, (), (),
    summary_funs = summary_funs, 
    latex = false, 
    grad_cutoff = 1e-4, 
    heywood_cutoff = 1e-4).data

# Set q to what is to be displayed in plot: 
# Figure  2: q = 3
# q = 3 
# Figure S1: q = 5
# q = 5 
# Figure S2: q = 8 
q = 8
simul_reduction = filter(x -> x.Model == Symbol("A_$q") || x.Model == Symbol("B_$q"), simul_reduction)
sort!(simul_reduction, [:Model, :n, :config])
grouped_data = groupby(simul_reduction, :Model)
ordered_models = first.(keys(sort(grouped_data.keymap)))
ordered_models = latexstring.(["{$(m[1])}_{$(m[3])}" for m in string.(first.(keys(sort(grouped_data.keymap))))]) 
configs = unique(simul_reduction.config)[[1, 2, 4, 3, 5]]

col = collect(cgrad(:viridis, length(configs), categorical  = true))
figsize = (1000, 500)
fontscale = 1.
fig = Figure(size = figsize, fontsize = 14 * fontscale)

side = nothing 
log_val = true 
group = :n 

labels = [
    L"\textrm{None}",
    L"\textrm{Akaike}[n]",
    L"\textrm{Hirose}[n]",
    L"\textrm{Akaike}[n^{-1/2}]",
    L"\textrm{Hirose}[n^{-1/2}]",
]

# Bias
bdf = DataFrame()
for (i, gdf) in enumerate(grouped_data)
    ns = log.(unique(gdf.n))
    grouped_data_n = groupby(gdf, :n)
    bias_df = DataFrame()
    for (j, gdfn) in enumerate(grouped_data_n)
        bias_tab_long_tmp = gdfn[!, ["Bias", "config"]]
        bias_tab = unstack(bias_tab_long_tmp, :config, :Bias)
        bias_tab = second_element.(bias_tab) 
        bias_tab = DataFrame(foldl(hcat, [vcat(bias_tab[1, col]...) for col in names(bias_tab)]), names(bias_tab))
        bias_tab = coalesce.(bias_tab, NaN)
        bias_tab_long = stack(bias_tab, names(bias_tab))
        bias_tab_long[!,"config"] = convert.(Int64, map(x -> findfirst(y -> y == x, configs), bias_tab_long.variable))
        bias_tab_long[!,"n"] .= unique(gdfn.n)
        append!(bias_df, bias_tab_long)
    end
    unique_ns = unique(bias_df.n)
    unique_configs = unique(bias_df.config)
    nconfig = length(unique_configs)
    dodge_width = group == :config ? 0.205 : 0.34
    x_positions = []
    if group == :config 
        for n in unique_ns
            base_x = n  
            for (i, config) in enumerate(unique_configs)
                push!(x_positions, (base_x + (i - mean(1:nconfig)) * dodge_width, n, config))
            end
        end
    else
        for conf in 1:length(unique_configs)
            base_x = conf  
            for (i, n) in enumerate(unique(gdf.n))
                push!(x_positions, (base_x + (i - mean(1:3)) * dodge_width, conf, n))
            end
        end
    end 

    xs = map(x -> x[1], x_positions) 
    (j,k) = Tuple(CartesianIndices(fill(NaN,1,2))[i])
    bias_df[!,"n"] = map(x -> findfirst(y -> y == x, sort(unique(bias_df.n))), bias_df.n)

    if !isnothing(side)
        bias_df = filter(row -> row.n != 2, bias_df)
        bias_df[!,"side"] = map(x -> x == 1 ? :left : :right, bias_df.n)
    end 
    if isnothing(side)
        if group == :config 
            ax = Axis(fig[j+1, k], xlabel = L"n", xticks = (1:3, latexstring.(unique(gdf.n))), title = ordered_models[i]) 
        else 
            ax = Axis(fig[j+1, k], xlabel = L"n", title = ordered_models[i]) 
        end 
        apply_fontscale!(ax, fontscale)
        if !log_val
            CairoMakie.ylims!(ax, [-0.05,0.05])
            vals = bias_df.value
            if group == :config 
                gbdf = groupby(bias_df, [:n, :variable])
                ms = map(x -> Statistics.mean(x.value), collect(gbdf)) 
            else
                gbdf = groupby(bias_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> Statistics.mean(x.value), collect(gbdf)) 
            end 
        else
            CairoMakie.ylims!(ax, -15,5)
            ax.yticks = (collect(-15:5:5), latexstring.(collect(-15:5:5)))
            vals = log.(abs.(bias_df.value))
            if group == :config 
                gbdf = groupby(bias_df, [:n, :variable])
                ms = map(x -> Statistics.mean(log.(abs.(x.value))), collect(gbdf)) 
            else
                gbdf = groupby(bias_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> Statistics.mean(log.(abs.(x.value))), collect(gbdf)) 
            end 
            
        end 
        if group == :config 
            CairoMakie.violin!(ax, bias_df.n, vals,  
                dodge = isnothing(side) ? bias_df.config : fill(1,nrow(bias_df)), 
                side = isnothing(side) ? nothing : bias_df.side,
                color = col[bias_df.config], 
                show_median=false, 
                gap= 0.01)

        else         
            CairoMakie.violin!(ax, bias_df.config, vals,  
                dodge = isnothing(side) ? bias_df.n : fill(1,nrow(bias_df)), 
                side = isnothing(side) ? :both : bias_df.side,
                color = col[bias_df.config], 
                show_median=false, 
                gap= 0.01
            )
        end 

        CairoMakie.scatter!(ax, xs, ms, color = :black)

        hidexdecorations!(ax, grid = false)
        if i > 1
            hideydecorations!(ax, grid = false)
        end 
    else
        ax = Axis(fig[j+1, k], xticks = (6:10, fill("", 5)), xlabel = ordered_models[i]) 
        apply_fontscale!(ax, fontscale)
        CairoMakie.ylims!(ax, [-15,5])
        violin!(ax, bias_df.config, log.(abs.(bias_df.value)),
        side = bias_df.side,
        color = col[bias_df.config], 
        gap = 0.01)
    end 
    bias_df[!,"model"] .= i
    append!(bdf, bias_df)
end 
bias_lab = L"\log(|\textrm{Bias}|)" 
Label(
    fig[2, 0],  
    bias_lab,    
    fontsize = 18 * fontscale,
    rotation = π/2,  
    halign = :center,
    valign = :center,
    padding = (0, 0, 0, 0),
    tellheight = false 
)
f = Figure() 
ax = f[1, 1] = Axis(f)
apply_fontscale!(ax, fontscale) 
elements = [PolyElement(polycolor = col[i]) for i in 1:length(labels)]
title = L"\textbf{\textrm{Penalty}}"
fig[1, :] = Legend(fig, elements, labels, 
    #title,
    framevisible = false,
    orientation = :horizontal, 
    labelsize = 14 * fontscale
)

# RMSE 
log_val = true 
bdf = DataFrame()
for (i, gdf) in enumerate(deepcopy(grouped_data))
    ns = log.(unique(gdf.n))
    grouped_data_n = groupby(gdf, :n)
    RMSE_df = DataFrame()
    for (j, gdfn) in enumerate(deepcopy(grouped_data_n))
        RMSE_tab_long_tmp = deepcopy(gdfn[!, ["RMSE", "config"]])
        #RMSE_tab_long = DataFrame(RMSE_tab_long_tmp)[[2, 1 , 5, 3, 4],:]
        RMSE_tab = unstack(RMSE_tab_long_tmp, :config, :RMSE)
        RMSE_tab = second_element.(RMSE_tab) 
        RMSE_tab = DataFrame(foldl(hcat, [vcat(RMSE_tab[1, col]...) for col in names(RMSE_tab)]), names(RMSE_tab))
        RMSE_tab = coalesce.(RMSE_tab, NaN)
        RMSE_tab_long = stack(RMSE_tab, names(RMSE_tab))
        RMSE_tab_long[!,"config"] = convert.(Int64, map(x -> findfirst(y -> y == x, unique(RMSE_tab_long.variable)[[1, 2, 4, 3, 5]]), RMSE_tab_long.variable))
        RMSE_tab_long[!,"n"] .= unique(gdfn.n)
        append!(RMSE_df, RMSE_tab_long)
    end
    unique_ns = unique(RMSE_df.n)
    unique_configs = unique(RMSE_df.config)
    nconfig = length(unique_configs)
    dodge_width = group == :config ? 0.205 : 0.34
    x_positions = []
    if group == :config 
        for n in unique_ns
            base_x = n  
            for (i, config) in enumerate(unique_configs)
                push!(x_positions, (base_x + (i - mean(1:nconfig)) * dodge_width, n, config))
            end
        end
    else
        for conf in 1:length(unique_configs)
            base_x = conf  
            for (i, n) in enumerate(unique(gdf.n))
                push!(x_positions, (base_x + (i - mean(1:3)) * dodge_width, conf, n))
            end
        end
    end
    xs = map(x -> x[1], x_positions) 
    (j,k) = Tuple(CartesianIndices(fill(NaN,1,2))[i])
    RMSE_df[!,"n"] = map(x -> findfirst(y -> y == x, sort(unique(RMSE_df.n))), RMSE_df.n)

    if !isnothing(side)
        RMSE_df = filter(row -> row.n != 2, RMSE_df)
        RMSE_df[!,"side"] = map(x -> x == 1 ? :left : :right, RMSE_df.n)
    end 
    if isnothing(side)
        if group == :config 
            ax = Axis(fig[j+2, k], xlabel = L"n", xticks = (1:3, latexstring.(unique(gdf.n)))) 
        else 
            ax = Axis(fig[j+2, k], xlabel = L"n", 
                #xticks = (xs, latexstring.(repeat([50, 100, 400], outer = length(labels)))), 
                ) 
        end 
        # text!(
        #     ax,
        #     [Point2f(0, 1)];
        #     text  = [ordered_models[i]],
        #     align = (:left, :top),
        #     space = :relative,
        #     offset    = (4, -4),        
        #     fontsize  = 16 * fontscale
        # )
        apply_fontscale!(ax, fontscale)
        if !log_val
            CairoMakie.ylims!(ax, [-0.05,0.3])
            ax.yticks = (0:0.1:0.3, latexstring.(collect(0:.1:.3)))
            vals = RMSE_df.value
            if group == :config 
                gbdf = groupby(RMSE_df, [:n, :variable])
                ms = map(x -> mean(x.value), collect(gbdf)) 
            else
                gbdf = groupby(RMSE_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> mean(x.value), collect(gbdf)) 
            end 
        else
            CairoMakie.ylims!(ax, [-7,0])
            ax.yticks = (0:-2:-8, latexstring.(collect(0:-2:-8)))
            vals = log.(abs.(RMSE_df.value))
            if group == :config 
                gbdf = groupby(RMSE_df, [:n, :variable])
                ms = map(x -> mean(log.(abs.(x.value))), collect(gbdf)) 
            else
                gbdf = groupby(RMSE_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> mean(log.(abs.(x.value))), collect(gbdf)) 
            end 
        end 
        if group == :config 
            violin!(ax, RMSE_df.n, vals,  
                dodge = isnothing(side) ? RMSE_df.config : fill(1,nrow(RMSE_df)), 
                side = isnothing(side) ? :both : RMSE_df.side,
                color = col[RMSE_df.config], 
                show_median=false, 
                # datalimits = extrema, 
                gap= 0.01)

        else         
            violin!(ax, RMSE_df.config, vals,  
                dodge = isnothing(side) ? RMSE_df.n : fill(1,nrow(RMSE_df)), 
                side = isnothing(side) ? :both : RMSE_df.side,
                color = col[RMSE_df.config], 
                show_median=false, 
                # datalimits = extrema, 
                gap= 0.01
            )
        end 
        scatter!(ax, xs, ms, color = :black)
        hidexdecorations!(ax, grid = false)
        if i > 1
            hideydecorations!(ax, grid = false)
        end 
    else
        ax = Axis(fig[j+3, k], xticks = (6:10, fill("", 5)), xlabel = ordered_models[i]) 
        apply_fontscale!(ax, fontscale)
        CairoMakie.ylims!(ax, [-0.05,0.3])
        violin!(ax, RMSE_df.config, log.(abs.(RMSE_df.value)),
        side = RMSE_df.side,
        color = col[RMSE_df.config], 
        gap = 0.01)
    end 
    RMSE_df[!,"model"] .= i
    append!(bdf, RMSE_df)
end 
RMSE_lab = L"\log(\textrm{RMSE})"
Label(
    fig[3, 0],  
    RMSE_lab,    
    fontsize = 18 * fontscale,
    rotation = π/2,  
    halign = :center,
    valign = :center,
    padding = (0, 0, 0, 0),
    tellheight = false 
)


# PU
log_val = false 
bdf = DataFrame()
for (i, gdf) in enumerate(deepcopy(grouped_data))
    ns = log.(unique(gdf.n))
    grouped_data_n = groupby(gdf, :n)
    PU_df = DataFrame()
    for (j, gdfn) in enumerate(deepcopy(grouped_data_n))
        PU_tab_long_tmp = deepcopy(gdfn[!, ["PU", "config"]])
        PU_tab = unstack(PU_tab_long_tmp, :config, :PU)
        PU_tab = second_element.(PU_tab) 
        PU_tab = DataFrame(foldl(hcat, [vcat(PU_tab[1, col]...) for col in names(PU_tab)]), names(PU_tab))
        PU_tab = coalesce.(PU_tab, NaN)
        PU_tab_long = stack(PU_tab, names(PU_tab))
        PU_tab_long[!,"config"] = convert.(Int64, map(x -> findfirst(y -> y == x, unique(PU_tab_long.variable)[[1, 2, 4, 3, 5]]), PU_tab_long.variable))
        PU_tab_long[!,"n"] .= unique(gdfn.n)
        append!(PU_df, PU_tab_long)
    end
    unique_ns = unique(PU_df.n)
    unique_configs = unique(PU_df.config)
    nconfig = length(unique_configs)
    dodge_width = group == :config ? 0.205 : 0.34
    x_positions = []
    if group == :config 
        for n in unique_ns
            base_x = n  
            for (i, config) in enumerate(unique_configs)
                push!(x_positions, (base_x + (i - mean(1:nconfig)) * dodge_width, n, config))
            end
        end
    else
        for conf in 1:length(unique_configs)
            base_x = conf  
            for (i, n) in enumerate(unique(gdf.n))
                push!(x_positions, (base_x + (i - mean(1:3)) * dodge_width, conf, n))
            end
        end
    end
    xs = map(x -> x[1], x_positions) 
    (j,k) = Tuple(CartesianIndices(fill(NaN,1,2))[i])
    PU_df[!,"n"] = map(x -> findfirst(y -> y == x, sort(unique(PU_df.n))), PU_df.n)

    if !isnothing(side)
        PU_df = filter(row -> row.n != 2, PU_df)
        PU_df[!,"side"] = map(x -> x == 1 ? :left : :right, PU_df.n)
    end 
    if isnothing(side)
        if group == :config 
            ax = Axis(fig[j+3, k], xlabel = L"n", 
                        xticks = (1:3, latexstring.(unique(gdf.n)))) 
        else 
            ax = Axis(fig[j+3, k], xlabel = L"n", 
                xticks = (xs, latexstring.(repeat(unique(gdf.n), outer = length(labels))))) 
        end 
    #    text!(
    #         ax,
    #         [Point2f(0, 1)];
    #         text  = [ordered_models[i]],
    #         align = (:left, :top),
    #         space = :relative,
    #         offset    = (4, -4),        
    #         fontsize  = 16 * fontscale
    #     )
        apply_fontscale!(ax, fontscale)
        if !log_val
            CairoMakie.ylims!(ax, [0.0,1.0])
            ax.yticks = (0:.25:1, latexstring.(collect(0:.25:1)))
            vals = PU_df.value
            if group == :config 
                gbdf = groupby(PU_df, [:n, :variable])
                ms = map(x -> mean(x.value), collect(gbdf)) 
            else
                gbdf = groupby(PU_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> mean(x.value), collect(gbdf)) 
            end 
        else
            CairoMakie.ylims!(ax, [-7,0])
            vals = log.(abs.(PU_df.value))
            if group == :config 
                gbdf = groupby(PU_df, [:n, :variable])
                ms = map(x -> mean(log.(abs.(x.value))), collect(gbdf)) 
            else
                gbdf = groupby(PU_df, [:variable, :n])
                inds = collect(1:length(gbdf))
                perm = [1, 2, 4, 3, 5]
                indscopy = copy(inds)
                for i in 1:Int(size(gbdf, 1) / length(ns))
                    inds[(i-1)*length(ns)+1:length(ns)*i] .= indscopy[(perm[i]-1)*length(ns)+1:length(ns)*perm[i]]
                end
                gbdf = gbdf[inds]
                ms = map(x -> mean(log.(abs.(x.value))), collect(gbdf)) 
            end 
        end 
        if group == :config 
            violin!(ax, PU_df.n, vals,  
                dodge = isnothing(side) ? PU_df.config : fill(1,nrow(PU_df)), 
                side = isnothing(side) ? :both : PU_df.side,
                color = col[PU_df.config], 
                show_median=false, 
                # datalimits = extrema, 
                gap= 0.01)

        else         
            violin!(ax, PU_df.config, vals,  
                dodge = isnothing(side) ? PU_df.n : fill(1,nrow(PU_df)), 
                side = isnothing(side) ? :both : PU_df.side,
                color = col[PU_df.config], 
                show_median=false, 
                # datalimits = extrema, 
                gap= 0.01
            )
        end 
        scatter!(ax, xs, ms, color = :black)
        lines!(ax, xs, fill(0.5, length(xs)), color = :gray, linestyle = :dash)
        if i > 1
            hideydecorations!(ax, grid = false)
        end 
    else
        ax = Axis(fig[j+3, k], xticks = (6:10, fill("", 5)), xlabel = ordered_models[i]) 
        apply_fontscale!(ax, fontscale)
        CairoMakie.ylims!(ax, [-0.05,0.3])
        violin!(ax, PU_df.config, log.(abs.(PU_df.value)),#log.(abs.(PU_df.value)), 
        side = PU_df.side,
        color = col[PU_df.config], 
        # datalimits = extrema, 
        gap = 0.01)
    end 
    PU_df[!,"model"] .= i
    append!(bdf, PU_df)
end 
PU_lab = L"\textrm{PU}"
Label(
    fig[4, 0],  
    PU_lab,    
    fontsize = 18 * fontscale,
    rotation = π/2,  
    halign = :center,
    valign = :center,
    padding = (0, 0, 0, 0),
    tellheight = false 
)
display(fig)
save(joinpath(figures_path, "violin-q$q.pdf"), fig)

