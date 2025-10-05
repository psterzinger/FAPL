module FactorModelSimul

using LinearAlgebra, DataFrames, PrettyTables, Distributions, Distributed, Random, Optim, Latexify, LaTeXStrings

include("FactorModels.jl") 
using .FactorModels

import Base.show 

export factor_model_simul, reduce_simul, FMsimul, PrettyDataFrame  

struct FMsimul 
    results::DataFrame 
    summary::Union{NamedTuple, DataFrame, Nothing} 
    models::NamedTuple 
    penalties::Tuple
    scalings::Tuple 
    ns::Union{Vector,Int64} 
    reps::Int64 
    options::Any 
    model_select::Bool
end 

struct PrettyDataFrame
    data::DataFrame
    args::Tuple 
    kwargs::Union{Nothing, NamedTuple}
end

function Base.show(io::IO, mime::MIME"text/plain", pdf::PrettyDataFrame)
    pretty_table(io, pdf.data, pdf.args...;pdf.kwargs...) 
end

function Base.show(io::IO, mime::MIME"text/plain", obj::FMsimul)
    if !isnothing(obj.summary)
        summary = obj.summary
        if isa(summary, NamedTuple) 
            hlines = summary.hlines 
            subheaders = summary.subheaders
            summary = summary.summary 
        else 
            hlines = nothing 
            subheaders = nothing 
        end 
        if !obj.model_select
            pretty_table(summary;
                header = (names(summary), subheaders),
                highlighters = (smallest_val_highlighter, smallest_val_highlighter2, largest_val_highlighter, min_PU_highlighter, min_PU_highlighter2), 
                formatters = round_formatter, 
                body_hlines = hlines) 
        else 
            pretty_table(summary;
                header = (names(summary), subheaders),
                highlighters = (max_val_HL1, max_val_HL2, min_val_HL1, min_val_HL2, min_quantile_HL), 
                formatters = round_formatter, 
                body_hlines = hlines) 
        end 
    else
        simul_summary(
        obj.results, obj.models; 
        excl_heywood = true, 
        excl_no_conv = true, 
        excl_grad_nz = true, 
        grad_cutoff = 1e-5, 
        heywood_cutoff = 1e-5,
        model_select = obj.model_select
        )
    end 
end

function factor_model_simul(models::NamedTuple, penalties::Tuple, scalings::Tuple, ns, reps; 
    method = :both, 
    optimizer = Optim.Newton(), 
    abstol = 1e-9, 
    reltol = 1e-9, 
    max_iter = 1000, 
    max_iter_EM = 100, 
    factor_method = :Bartlett, 
    model_select = false,
    seed_base = 0,
    include_MLE = true)  
    results = []
    for (model_name, (lambda, psi)) in zip(keys(models), models) 
        (p, q) = size(lambda)
        npar = q * (p + 1)
        sigma = lambda * lambda' + psi 
        d = MvNormal(zeros(p), sigma) 
        for n in ns
            X = fill(NaN, (n, p)) 
            temp_results = pmap(1:reps) do i
                Random.seed!(seed_base + i)
                X .= rand(d, n)' 
                rep_results = []
                fmfit = nothing
                for scaling in scalings 
                    for (lambda_pen, psi_pen, comp_pen) in penalties 
                        try
                            if model_select
                                fmfit = fit_factor_model(X; comp_pen = getfield(Main, comp_pen), psi_pen = getfield(Main, psi_pen), lambda_pen = getfield(Main, lambda_pen), method = method, optimizer = optimizer, scaling = getfield(Main, scaling), abstol = abstol, reltol = reltol, max_iter = max_iter, max_iter_EM = max_iter_EM, factor_method = factor_method)
                            else 
                                fmfit = fit_factor_model(X, q; comp_pen = getfield(Main, comp_pen), psi_pen = getfield(Main, psi_pen), lambda_pen = getfield(Main, lambda_pen), method = method, optimizer = optimizer, scaling = getfield(Main, scaling), abstol = abstol, reltol = reltol, max_iter = max_iter, max_iter_EM = max_iter_EM, factor_method = factor_method)
                            end 
                        catch e 
                            @error "Penalised" exception=(e, catch_backtrace())
                            options = (scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, method = method, optimizer = optimizer, max_iter = max_iter, abstol = abstol, reltol = reltol, max_iter_EM = max_iter_EM, factor_method = factor_method)
                            if model_select
                                fmfit = FactorModels.FMfit(X, (fill(NaN, length(npar)), fill(NaN, length(npar))), (fill(NaN, (p, q)), fill(NaN, (p, q))), (Diagonal(fill(NaN, p)), Diagonal(fill(NaN, p))), (fill(NaN, (p, p)), fill(NaN, (p, p))), (), (false, false), NaN, NaN, (fill(NaN, length(npar)), fill(NaN, length(npar))), fill(NaN, (npar, npar)), (NaN, NaN), (NaN, NaN), models, options)
                            else 
                                fmfit = FactorModels.FMfit(X, fill(NaN, length(npar)), fill(NaN, (p, q)), Diagonal(fill(NaN, p)), fill(NaN, (p, p)), fill(NaN, (n, q)), false, NaN, NaN, fill(NaN, length(npar)), fill(NaN, (npar, npar)), q, NaN, models, options)
                            end
                        end 
                        push!(rep_results, (fmfit, n, model_name, Tuple(filter(x -> !isnothing(getfield(Main,x)), (lambda_pen, psi_pen, comp_pen))), scaling, seed_base + i))
                    end
                end
                if include_MLE
                    try 
                        if model_select
                            fmfit = fit_factor_model(X; comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, method = method, optimizer = optimizer, scaling = [0.,0.], abstol = abstol, reltol = reltol, max_iter = max_iter, max_iter_EM = max_iter_EM, factor_method = factor_method) 
                        else
                            fmfit = fit_factor_model(X, q; comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, method = method, optimizer = optimizer, scaling = [0.,0.], abstol = abstol, reltol = reltol, max_iter = max_iter, max_iter_EM = max_iter_EM, factor_method = factor_method) 
                        end 
                    catch e 
                        @error "Unpenalised" exception=(e, catch_backtrace())
                        options = (scaling = [0,0], comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, method = method, optimizer = optimizer, max_iter = max_iter, abstol = abstol, reltol = reltol, max_iter_EM = max_iter_EM, factor_method = factor_method)
                        if model_select
                            fmfit = FactorModels.FMfit(X, (fill(NaN, length(npar)), fill(NaN, length(npar))), (fill(NaN, (p, q)), fill(NaN, (p, q))), (Diagonal(fill(NaN, p)), Diagonal(fill(NaN, p))), (fill(NaN, (p, p)), fill(NaN, (p, p))), (), (false, false), NaN, NaN, (fill(NaN, length(npar)), fill(NaN, length(npar))), fill(NaN, (npar, npar)),  (NaN, NaN), (NaN, NaN), models, options)
                        else 
                            fmfit = FactorModels.FMfit(X, fill(NaN, length(npar)), fill(NaN, (p, q)), Diagonal(fill(NaN, p)), fill(NaN, (p, p)), fill(NaN, (n, q)), false, NaN, NaN, fill(NaN, length(npar)), fill(NaN, (npar, npar)), q, NaN, models, options)
                        end
                    end
                    push!(rep_results, (fmfit, n, model_name, (), [0, 0], seed_base + i))  
                end 
                rep_results
            end 
            append!(results, temp_results)
        end
    end
    colnames = ["FMfit", "n", "model", "pen", "scaling", "seed"]
    df = DataFrame([[] for _ in colnames],colnames)
    for res in results
        for row in res
            push!(df, row)
        end
    end
    options = (method = method, optimizer = optimizer, abstol = abstol, reltol = reltol, max_iter = max_iter, max_iter_EM = max_iter_EM)
    summary, hlines, subheaders =  simul_summary(
        df, models; 
        excl_heywood = true, 
        excl_no_conv = true, 
        excl_grad_nz = true, 
        grad_cutoff = 1e-5,  
        print_only = false, 
        model_select = model_select
        )
    return FMsimul(df, (summary = summary, hlines = hlines, subheaders = subheaders), models, penalties, scalings, ns, reps, options, model_select) 
end 

function get_simul_pars(df; 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-6,
    heywood_cutoff = nothing) 
    fits =  df.FMfit 
    lambdas = map(x -> x.Lambda, fits)
    psis = map(x -> diag(x.Psi), fits)
    inds = fill(1.0, length(lambdas))
    if excl_grad_nz
        grads =  map(x -> x.grad_pen, fits) 
        hess = map(x -> x.hessian_pen, fits)
        sups = fill(Inf, length(fits)) 
        for i in eachindex(sups) 
            try 
                sups[i] = maximum(abs.(hess[i] \ grads[i])) 
            catch 
            end
        end 
        inds = inds .* (sups .< grad_cutoff)  
    end 
    if excl_heywood
            if isnothing(heywood_cutoff)
                hw = map(x -> x.Heywood, fits)
                inds = inds .* .!(.==(1).(hw)) 
            else
                hw = map(x -> any(diag(x * x') .* heywood_cutoff .> 1.0), lambdas) .| map(x -> any(x .* heywood_cutoff .> 1.0), psis) .|  map(x ->  any(x .< heywood_cutoff), psis)
                inds = inds .* .!(.==(1).(hw)) 
            end 
    end 
    if excl_no_conv 
        conv = map(x -> x.conv, fits)
        inds = inds .* conv 
    end 
    factor_scores = map(x -> x.factor_scores, fits) 
    return lambdas, psis, Bool.(inds), factor_scores
end 

function compute_MSE(thetas, theta_true)
    compute_bias(thetas, theta_true)^2 + compute_var(thetas, theta_true)
end

function compute_bias(thetas, theta_true)
    if isempty(thetas) 
        return NaN 
    end
    mean(mean.(map(x -> x .- theta_true, thetas)))
end

function compute_var(thetas, theta_true)
    if isempty(thetas) 
        return NaN 
    end
    mean(mean.(map(x -> ((x .- theta_true).^2), thetas)))
end

function compute_RMSE(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    # mean(map(x -> sqrt(mean((x - theta_true).^2)), thetas))
    sqrt(compute_MSE(thetas, theta_true)) 
end

function compute_rel_bias(thetas,theta_true) 
    ind = .!iszero.(theta_true) 
    mean(map(x -> mean((x[ind] - theta_true[ind]) ./ theta_true[ind]), thetas)) 
end 

function compute_abs_bias(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    mean(mean(map(x -> abs.(x - theta_true), thetas)))
end 

function compute_rel_RMSE(thetas, theta_true) 
    ind = .!iszero.(theta_true) 
    mean(map(x -> sqrt(mean(((x[ind] - theta_true[ind]) ./ theta_true[ind]).^2)), thetas))
end
 
function compute_PU(thetas, theta_true) 
    if isempty(thetas) 
        return NaN 
    end
    inds = map(x -> !any(isnan.(x)), thetas) 
    mean(mean.(map(x -> (x .- theta_true) .< 0.0 , thetas[inds])))
end

function get_FS(seed, n, q) 
    Random.seed!(seed) 
    randn(n, q) 
end 

function compute_summary(df, lambda_true, psi_true, summary_funs; 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-5, 
    heywood_cutoff = 1e-5)
    lambdas, psis, inds, factor_scores = get_simul_pars(df; excl_heywood = excl_heywood, excl_no_conv = excl_no_conv, excl_grad_nz = excl_grad_nz, grad_cutoff = grad_cutoff, heywood_cutoff = heywood_cutoff) 
    sum_names = collect(keys(summary_funs)) 
    sum_names = vcat(sum_names, :FS_MSE, Symbol("% used"), Symbol("% Heywood"))
    sumdf = DataFrame([[] for _ in sum_names], sum_names) 
    sums = []
    for sum_fun in summary_funs
        push!(sums, (sum_fun(map(x -> varimax(x), lambdas[inds]), varimax(lambda_true)), sum_fun(map(x -> lt(x * x'), lambdas[inds]), lt(lambda_true * lambda_true')), sum_fun(psis[inds], diag(psi_true))))
    end 
    if sum(inds) > 0
        n, p = size(df.FMfit[1].X) 
        q = df.FMfit[1].q 
        true_factor_scores = [Matrix{Float64}(undef, n, q) for _ in 1:sum(inds)] 
        count = 1 
        d = MvNormal(zeros(q), inv(lambda_true' * inv(psi_true) * lambda_true + I(q)))
        for ind in findall(inds) 
            Random.seed!(df.seed[ind]) 
            true_factor_scores[count] .= rand(d, n)' 
            count += 1 
        end 
        FS_diff = vcat(factor_scores[inds]...) - vcat(true_factor_scores...)
        FS_diff = FS_diff.^2
        push!(sums, mean(FS_diff)) 
    else 
        push!(sums, NaN) 
    end 
    push!(sums, sum(inds[.!isnan.(inds)]) / length(lambdas)) 
    if !isnothing(heywood_cutoff)
        hw = map(x -> any(diag(x * x') .* heywood_cutoff .> 1.0), lambdas) .| map(x -> any(x .* heywood_cutoff .> 1), psis) .|  map(x ->  any(x .< heywood_cutoff), psis)
    else
        hw = map(x -> isnan(x.Heywood) ? 0 : x.Heywood, df.FMfit)
    end 
    push!(sums, sum(hw) / length(lambdas))
    push!(sumdf, sums)

    sumdf
end 

function lt(A) 
    A[tril!(trues(size(A)), 0)]
end 

# function simul_summary(df, models,
#     summary_funs = (Bias = compute_bias, 
#     Abs_Bias = compute_abs_bias,
#     RMSE = compute_RMSE, 
#     PU = compute_PU); 
#     excl_heywood = true, 
#     excl_no_conv = true, 
#     excl_grad_nz = true, 
#     grad_cutoff = 1e-6, 
#     heywood_cutoff = 1e-5, 
#     model_select = false, 
#     print_only = true
#     )
#     sort!(df, [:model, :n])
#     grouped_m = groupby(df, [:model])
#     if model_select 
#         return simul_summary_ms(df, models; 
#         excl_heywood = excl_heywood, 
#         excl_no_conv = excl_no_conv, 
#         excl_grad_nz = excl_grad_nz, 
#         grad_cutoff = grad_cutoff, 
#         print_only = print_only, 
#         heywood_cutoff = heywood_cutoff)
#     end 
#     outdf = DataFrame([[] for _ in 1:(length(summary_funs) + 6)], :auto)
#     count = 0 
#     body_hlines = Int64[] 
#     for g in grouped_m 
#         model = g.model[1] 
#         lambda_true, psi_true = getfield(models, model) 
#         grouped_n = groupby(g, [:n])
#         push!(outdf, vcat(model, ["" for _ in 1:(length(names(outdf))-1)]); promote = true)
#         count += 1  
#         push!(body_hlines, count) 
#         for gn in grouped_n 
#             n = gn.n[1] 
#             configs = unique(gn[!,2:end-1])
#             sumdf = DataFrame()
#             for config in eachrow(configs) 
#                 lambda_true, psi_true = getfield(models, model)
#                 temp_sumdf = compute_summary(filter(row -> row[2:end-1] == config, gn), lambda_true, psi_true, summary_funs, excl_heywood = excl_heywood, excl_no_conv = excl_no_conv, excl_grad_nz = excl_grad_nz, grad_cutoff = grad_cutoff, heywood_cutoff = heywood_cutoff)   
#                 concat_config = join([string(k, "=", v) for (k, v) in pairs(config[3:4])], "; ")
#                 temp_sumdf[!, :config] = repeat([concat_config], nrow(temp_sumdf))
#                 temp_sumdf = insertcols!(temp_sumdf, 1, :n => "")
#                 temp_sumdf = insertcols!(temp_sumdf, 1, :Model => "")
#                 append!(sumdf, temp_sumdf)
#                 count += nrow(temp_sumdf) 
#             end 
#             pushfirst!(sumdf, vcat("", n, ["" for _ in 1:ncol(sumdf)-2]); promote = true)
#             count += 1 
#             push!(body_hlines, count) 
#             rename!(outdf,names(sumdf)) 
#             append!(outdf, sumdf) 
#         end 
#     end 
#     if print_only
#         subheaders = vcat(fill("", 2), fill("(ΛO, ΛΛ', Ψ)", length(summary_funs)), "", fill("%", 2), "")
#         pretty_table(outdf;
#             header = (names(outdf), subheaders),
#             highlighters = (smallest_val_highlighter, smallest_val_highlighter2, largest_val_highlighter, min_PU_highlighter, min_PU_highlighter2), 
#             formatters = round_formatter, 
#             body_hlines = body_hlines) 
#     else 
#         return outdf, body_hlines 
#     end 
# end 

function simul_summary(df, models,
    summary_funs = (Bias = compute_bias, 
    Abs_Bias = compute_abs_bias,
    RMSE = compute_RMSE, 
    PU = compute_PU); 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-6, 
    heywood_cutoff = 1e-5, 
    model_select = false, 
    print_only = true)
    sort!(df, [:model, :n])
    grouped_m = groupby(df, [:model])
    if model_select 
        return simul_summary_ms(df, models; 
            excl_heywood = excl_heywood, 
            excl_no_conv = excl_no_conv, 
            excl_grad_nz = excl_grad_nz, 
            grad_cutoff = grad_cutoff, 
            print_only = print_only, 
            heywood_cutoff = heywood_cutoff
        )
    end 
    outdf = DataFrame([[] for _ in 1:(length(summary_funs) + 6)], :auto)
    count = 0 
    body_hlines = Int64[] 
    for g in grouped_m 
        model = g.model[1] 
        lambda_true, psi_true = getfield(models, model) 
        grouped_n = groupby(g, [:n])
        push!(outdf, vcat(model, ["" for _ in 1:(length(names(outdf))-1)]); promote = true)
        count += 1  
        push!(body_hlines, count) 
        for gn in grouped_n 
            n = gn.n[1] 
            grouped_c = groupby(gn, [:pen, :scaling])
            configs = map(x -> NamedTuple(x), keys(grouped_c)) 
            sumdf = DataFrame()
            for (config, gc) in zip(configs, grouped_c)
                temp_sumdf = compute_summary(gc, lambda_true, psi_true, summary_funs, excl_heywood = excl_heywood, excl_no_conv = excl_no_conv, excl_grad_nz = excl_grad_nz, grad_cutoff = grad_cutoff, heywood_cutoff = heywood_cutoff)   
                temp_sumdf[!, :config] = repeat([sprint(show, config) ], nrow(temp_sumdf))
                temp_sumdf = insertcols!(temp_sumdf, 1, :n => "")
                temp_sumdf = insertcols!(temp_sumdf, 1, :Model => "")
                append!(sumdf, temp_sumdf)
                count += nrow(temp_sumdf) 
            end 
            pushfirst!(sumdf, vcat("", n, ["" for _ in 1:ncol(sumdf)-2]); promote = true)
            count += 1 
            push!(body_hlines, count) 
            rename!(outdf,names(sumdf)) 
            append!(outdf, sumdf) 
        end 
    end 
    subheaders = vcat(fill("", 2), fill("(ΛO, ΛΛ', Ψ)", length(summary_funs)), "", fill("%", 2), "")
    if print_only
        pretty_table(outdf;
            header = (names(outdf), subheaders),
            highlighters = (smallest_val_highlighter, smallest_val_highlighter2, largest_val_highlighter, min_PU_highlighter, min_PU_highlighter2), 
            formatters = round_formatter, 
            body_hlines = body_hlines) 
    else 
        return outdf, body_hlines, subheaders
    end 
end 

function simul_summary_ms(df, models;
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9], 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-6, 
    print_only = true, 
    heywood_cutoff = 1e-5
    )

    grouped_m = groupby(df, [:model])
    
    headers = ["Model", "n", "Correct AIC", "AIC quantiles", "Correct BIC", "BIC quantiles", "% used", "% Heywood", "config"]
    subheaders = vcat(fill("", 2), repeat(["%", Tuple(string.(quantiles))], outer = 2), fill("%", 2), "")
    outdf = DataFrame([[] for _ in 1:9], headers)

    count = 0 
    body_hlines = Int64[] 
    for g in grouped_m 
        model = g.model[1] 
        lambda_true, psi_true = getfield(models, model) 
        grouped_n = groupby(g, [:n])
        push!(outdf, vcat(model, ["" for _ in 1:(length(names(outdf))-1)]); promote = true)
        count += 1  
        push!(body_hlines, count) 
        for gn in grouped_n 
            n = gn.n[1] 
            configs = unique(gn[!,2:end-1])
            sumdf = DataFrame([[] for _ in 1:9], headers)
            for config in eachrow(configs) 
                lambda_true, psi_true = getfield(models, model)
                q_true = size(lambda_true, 2) 
                filter_df = filter(row -> row[2:end-1] == config, gn)
                fits = filter_df.FMfit

                q_AIC = map(x -> x.q[1], fits)
                q_BIC = map(x -> x.q[2], fits) 

                inds_AIC = fill(1, length(q_AIC)) 
                inds_BIC = fill(1, length(q_BIC)) 
                if isnothing(heywood_cutoff)
                    hey_AIC = map(x -> isnan(x.Heywood[1]) ? 0 : x.Heywood[1], fits)
                    hey_BIC = map(x -> isnan(x.Heywood[2]) ? 0 : x.Heywood[2], fits) 
                else 
                    lambdas_AIC = map(x -> x.Lambda[1], fits)
                    psis_AIC = map(x -> any(isnan.(x.Psi[1])) ? NaN : diag(x.Psi[1]), fits)
                    lambdas_BIC = map(x -> x.Lambda[2], fits)
                    psis_BIC = map(x -> any(isnan.(x.Psi[2])) ? NaN : diag(x.Psi[2]), fits)
                    nan_AIC = map(x -> any(isnan.(x)), lambdas_AIC) 
                    nan_BIC = map(x -> any(isnan.(x)), lambdas_BIC)  
                    hey_AIC = fill(0, length(lambdas_AIC)) 
                    hey_BIC = fill(0, length(lambdas_BIC))  
                    hey_AIC[.!nan_AIC] .= map(x -> any(diag(x * x') .* heywood_cutoff .> 1.0), lambdas_AIC[.!nan_AIC]) .| map(x -> any(x .* heywood_cutoff .> 1), psis_AIC[.!nan_AIC]) .|  map(x ->  any(x .< heywood_cutoff), psis_AIC[.!nan_AIC])
                    hey_BIC[.!nan_BIC] .= map(x -> any(diag(x * x') .* heywood_cutoff .> 1.0), lambdas_BIC[.!nan_BIC]) .| map(x -> any(x .* heywood_cutoff .> 1), psis_BIC[.!nan_BIC]) .|  map(x ->  any(x .< heywood_cutoff), psis_BIC[.!nan_BIC])
                end 
                perc_hw_AIC = sum(hey_AIC) / length(inds_AIC)
                perc_hw_BIC = sum(hey_BIC) / length(inds_BIC)    
                if excl_no_conv
                    conv_AIC = map(x -> x.conv[1], fits) 
                    conv_BIC = map(x -> x.conv[2], fits) 
                    inds_AIC = inds_AIC .* conv_AIC
                    inds_BIC = inds_BIC .* conv_BIC 
                end 

                if excl_grad_nz
                    grads_AIC = map(x -> x.grad_pen[1], fits) 
                    grads_BIC = map(x ->  x.grad_pen[2], fits) 
                    hess_AIC = map(x -> x.hessian_pen[1], fits)
                    hess_BIC = map(x -> x.hessian_pen[2], fits)
                    sups_AIC = fill(Inf, length(fits)) 
                    sups_BIC = similar(sups_AIC) 

                    for i in eachindex(sups_AIC) 
                        try 
                            sups_AIC[i] = maximum(abs.(hess_AIC[i] \ grads_AIC[i])) 
                        catch 
                        end
                        try 
                            sups_BIC[i] = maximum(abs.(hess_BIC[i] \ grads_BIC[i])) 
                        catch 
                        end
                    end 
                    inds_AIC = inds_AIC .* (sups_AIC .< grad_cutoff)  
                    inds_BIC = inds_BIC .* (sups_BIC .< grad_cutoff)  
                end
                inds_AIC = Bool.(inds_AIC) 
                inds_BIC = Bool.(inds_BIC) 
                #=
                if any(inds_AIC)
                    perc_hw_AIC = sum(hey_AIC[inds_AIC]) / length(inds_AIC)
                else
                    perc_hw_AIC = NaN 
                end 
                if any(inds_BIC)
                    perc_hw_BIC = sum(hey_BIC[inds_BIC]) / length(inds_BIC)    
                else
                    perc_hw_BIC = NaN 
                end 
                =#
                if excl_heywood 
                    inds_AIC .= inds_AIC .&& .!Bool.(hey_AIC)
                    inds_BIC .= inds_BIC .&& .!Bool.(hey_BIC)
                end 
               
                if !isempty(skipnan(q_AIC[inds_AIC]))
                    perc_corr_AIC = sum(map(x -> x == q_true, q_AIC[inds_AIC])) / sum(inds_AIC)
                    perc_used_AIC = sum(skipnan(inds_AIC)) / length(inds_AIC) 
                    quantiles_AIC = quantile(skipnan(q_AIC[inds_AIC] .- q_true), quantiles)
                else
                    quantiles_AIC = fill(NaN, length(quantiles)) 
                    perc_corr_AIC = NaN 
                    perc_used_AIC = 0 
                end 

                if !isempty(skipnan(q_BIC[inds_BIC]))
                    quantiles_BIC = quantile(skipnan(q_BIC[inds_BIC] .- q_true), quantiles) 
                    perc_corr_BIC = sum(map(x -> x == q_true, q_BIC[inds_BIC])) / sum(inds_BIC)
                    perc_used_BIC = sum(skipnan(inds_BIC)) / length(inds_BIC) 
                else 
                    quantiles_BIC = fill(NaN, length(quantiles)) 
                    perc_corr_BIC = NaN 
                    perc_used_BIC = 0
                end 
                concat_config = join([string(k, "=", v) for (k, v) in pairs(config[3:4])], "; ")
                row = ("", "", perc_corr_AIC, quantiles_AIC, perc_corr_BIC, quantiles_BIC, (perc_used_AIC, perc_used_BIC), (perc_hw_AIC, perc_hw_BIC), concat_config)
                push!(sumdf, row)
                count += 1
            end 
            pushfirst!(sumdf, vcat("", n, ["" for _ in 1:ncol(sumdf)-2]); promote = true)
            count += 1 
            push!(body_hlines, count) 
            rename!(outdf,names(sumdf)) 
            append!(outdf, sumdf) 
        end 
    end 
    subheaders = vcat(fill("", 2), repeat(["%", Tuple(string.(quantiles))], outer = 2), fill(("%", "%"), 2), "")
    if print_only
        pretty_table(outdf;
            header = (names(outdf), subheaders),
            highlighters = (max_val_HL1, max_val_HL2, min_val_HL1, min_val_HL2, min_quantile_HL), 
            formatters = round_formatter, 
            body_hlines = body_hlines) 
    else 
        return outdf, body_hlines, subheaders 
    end 
end 

function highlight_smallest_val(data, i, j, cols)
    return j ∈ cols && i ∈ block_mins(first_element.(data[:, j]))
end

function highlight_smallest_val2(data, i, j, cols)
    return j ∈ cols && i ∈ block_mins(second_element.(data[:, j]))
end

function highlight_smallest_val3(data, i, j, cols)
    return j ∈ cols && i ∈ block_mins(third_element.(data[:, j]))
end

function highlight_largest_val(data, i, j, cols)
    return j ∈ cols && i ∈ block_maxs(first_element.(data[:, j]))
end

function highlight_largest_val2(data, i, j, cols)
    return j ∈ cols && i ∈ block_maxs(second_element.(data[:, j]))
end

max_val_HL1 = Highlighter((data, i, j) ->  highlight_largest_val(data, i, j, [3, 5, 7]), foreground = :blue)
max_val_HL2 = Highlighter((data, i, j) ->  highlight_largest_val2(data, i, j, [7]), background = :blue) 
min_val_HL1 = Highlighter((data, i, j) ->  highlight_smallest_val(data, i, j, [8]), foreground = :blue)
min_val_HL2 = Highlighter((data, i, j) ->  highlight_smallest_val2(data, i, j, [8]), background = :blue)
min_quantile_HL = Highlighter((data, i, j) ->  highlight_smallest_val3(data, i, j, [4, 6]), foreground = :blue)

function first_element(x) 
    x isa Tuple ? first(x) : x
end 

function second_element(x) 
    x isa Tuple ? x[2] : x
end 

function third_element(x) 
    length(x) > 2 ? x[3] : x
end 

function find_block(v)
    starts = findfirst(x -> !isa(x, String), v) 
    if isnothing(starts) 
        return 
    else 
        ends =  findfirst(x -> isa(x, String), v[starts:end]) 
        if isnothing(ends) 
            ends = length(v)
        else 
            ends += starts - 2 
        end 
        return starts:ends 
    end  
end 

function find_blocks(v) 
    blocks = [] 
    block = find_block(v) 
    push!(blocks, block) 
    starts = last(block)
    while starts < length(v) 
        block = find_block(v[starts+1:end])
        if !isnothing(block) 
            block = block .+ starts 
            push!(blocks, block) 
            starts = last(block)
        else 
            break
        end 
    end 
    blocks 
end       

function skipnan(v) 
    v[.!isnan.(v)] 
end 

function block_mins(v)
    blocks = find_blocks(v) 
    argmins = fill(Int64[], length(blocks))
    for (i, block) in enumerate(blocks)
        min_inds = [] 
        try
            min = minimum(skipnan(abs.(v[block])))
            min_inds = findall(x -> abs(x) == min, v[block]) .+ first(block) .- 1 
        catch 
        end 
        argmins[i] = min_inds
    end 
    vcat(argmins...)
end

function block_maxs(v)
    blocks = find_blocks(v) 
    argmaxs = fill([], length(blocks))
    for (i, block) in enumerate(blocks)
        max_inds = []
        try
            max = maximum(skipnan(abs.(v[block]))) 
            max_inds = findall(x -> abs(x) == max, v[block]) .+ first(block) .- 1 
        catch 
        end 
        argmaxs[i] = max_inds
    end 
    vcat(argmaxs...)
end

function mincols(data) 
    inds = findall(values(mapcols(x -> all(isa.(x, Union{Tuple, Number, String})), data)[1,:]))
    n_ind = findfirst(x -> isequal(x, "n"), names(data)) 
    used_ind = findfirst(x -> occursin(r"used", x), names(data)) 
    config_ind = findfirst(x -> isequal(x, "config"), names(data)) 
    PU_ind = findfirst(x -> isequal(x, "PU"), names(data)) 
    setdiff(inds, vcat(n_ind, used_ind, config_ind, PU_ind)) 
end 

function PU_cols(data) 
    findfirst(x -> occursin(r"PU", x), names(data)) 
end 

function maxcols(data) 
    findfirst(x -> occursin(r"used", x), names(data)) 
end 

function highlight_smallest_val(data, i, j)
    return j ∈ mincols(data) && i ∈ block_mins(first_element.(data[:, j]))
end

function highlight_smallest_val2(data, i, j)
    return j ∈ mincols(data) && i ∈ block_mins(second_element.(data[:, j]))
end

function highlight_largest_val(data, i, j)
    return j ∈ maxcols(data) && i ∈ block_maxs(first_element.(data[:, j]))
end

function highlight_less_perc(data, i, j, perc, cols) 
    return j ∈ cols && i ∈ less_perc_inds(first_element.(data[:, j]), perc) 
end 

function highlight_betw_perc(data, i, j, lower, upper, cols) 
    return j ∈ cols && i ∈ betw_perc_inds(first_element.(data[:, j]), lower, upper)
end 

function highlight_PU(data, i, j) 
    if j ∈ PU_cols(data)
        dat = data[:,j] 
        inds = isa.(dat, Tuple) 
        dat[inds] .= first_element.(dat[inds]) .- 0.5 
        if i ∈ block_mins(dat) 
            return true 
        end 
    end 
    return false 
end 

function highlight_PU2(data, i, j) 
    if j ∈ PU_cols(data)
        dat = data[:,j] 
        inds = isa.(dat, Tuple) 
        dat[inds] .= second_element.(dat[inds]) .- 0.5 
        if i ∈ block_mins(dat) 
            return true 
        end 
    end 
    return false 
end 

function less_perc_inds(v, perc) 
    num_ind = isa.(v, Number) 
    vc = copy(v) 
    vc[.!num_ind] .= Inf 
    findall(x -> x < perc, vc) 
end 

function betw_perc_inds(v, lower, upper) 
    num_ind = isa.(v, Number) 
    vc = copy(v) 
    vc[.!num_ind] .= Inf 
    findall(x -> x < upper && x >= lower, vc) 
end 

function round_formatter(v, i, j; ndigits = 3) 
    if isa(v, Union{Tuple, Number}) 
        if isa(v[1], Vector)
            Tuple([round.(vec, digits = ndigits) for vec in v])
        else
            round.(v, digits = ndigits) 
        end 
    else
        v
    end 
end 

function forward_fill!(column)
    last_seen = nothing
    for i in eachindex(column)
        if column[i] == "" || column[i] === missing
            column[i] = last_seen
        else
            last_seen = column[i]
        end
    end
end

function replace_empty_with_missing!(df)
    for col in eachcol(df)
        replace!(col, "" => missing)
        replace!(col, NaN => missing)
        replace!(col, (NaN, NaN) => missing)
    end
end

function minimum_nan(v)
    inds = .!isnan.(v) 
    minimum(v[inds])
end 

smallest_val_highlighter = Highlighter(highlight_smallest_val, foreground = :blue, bold = true)
smallest_val_highlighter2 = Highlighter(highlight_smallest_val2, background = :blue, bold = true)
largest_val_highlighter = Highlighter(highlight_largest_val, foreground = :blue, bold = true)
min_PU_highlighter = Highlighter(highlight_PU, foreground = :blue, bold = true)
min_PU_highlighter2 = Highlighter(highlight_PU2, background = :blue, bold = true)
#five_perc_highlighter = Highlighter((data, i, j) -> highlight_less_perc(data, i, j, 0.05, findall(occursin.(r"rel_", names(data)))), background = :green)
#ten_perc_highlighter = Highlighter((data, i, j) -> highlight_betw_perc(data, i, j, 0.05, 0.1, findall(occursin.(r"rel_", names(data)))), background = :yellow)
#more_ten_perc_highlighter = Highlighter((data, i, j) -> highlight_betw_perc(data, i, j, 0.01, Inf, findall(occursin.(r"rel_", names(data)))), background = :red)

latex_smallest_val_highlighter = LatexHighlighter(highlight_smallest_val, ["color{blue}", "textbf"])
latex_smallest_val_highlighter2 = LatexHighlighter(highlight_smallest_val2, ["cellcolor{cyan}"])
latex_largest_val_highlighter = LatexHighlighter(highlight_largest_val, ["color{blue}", "textbf"])
latex_min_PU_highlighter = LatexHighlighter(highlight_PU, ["color{blue}", "textbf"])
latex_min_PU_highlighter2 = LatexHighlighter(highlight_PU2, ["cellcolor{cyan}"])

function reduce_simul(fmsimul, filters = (), selections = ();
    summary_funs = (Bias = compute_bias, 
        Abs_bias = compute_abs_bias, 
        RMSE = compute_RMSE, 
        Var = compute_var, 
        PU = compute_PU),
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-6, 
    heywood_cutoff = 1e-5, 
    latex = false )

    df = deepcopy(fmsimul.results)
    if length(filters) > 1
        filtered_df = foldl((df, f) -> filter(f, df), filters, init = df)
    elseif length(filters) == 1
        filtered_df = filter(first(filters), df) 
    else
        filtered_df = df 
    end 
    if isempty(filtered_df)
        @warn "DataFrame empty after filtering" 
        return 
    end 
    sort!(filtered_df, [:model, :n])
    outdf, body_hlines = simul_summary(filtered_df, deepcopy(fmsimul.models), summary_funs; 
        excl_heywood = excl_heywood, 
        excl_no_conv = excl_no_conv, 
        excl_grad_nz = excl_grad_nz, 
        grad_cutoff = grad_cutoff, 
        heywood_cutoff = heywood_cutoff,
        print_only = false, 
    )
    forward_fill!(outdf.n)
    forward_fill!(outdf.Model)
    replace_empty_with_missing!(outdf)
    outdf = dropmissing(outdf) 
    if !(selections == ())
        select_df = select(outdf, selections)
        if isempty(select_df)
            @warn "DataFrame empty after selection" 
            return 
        end 
    else 
        select_df = outdf 
    end 
    tuple_cols = findall(values(mapcols(v -> all(isa.(v, Union{String, Tuple})), select_df)[1,:])) 
    perc_cols = findall(occursin.(r"%", names(select_df))) 
    
    if latex 
        subheaders = fill(LatexCell(L" "), ncol(select_df)) 
        subheaders[tuple_cols] .= fill(LatexCell(L"(\Lambda \mathrm{O}, \Lambda \Lambda^{\top}, \Psi)"), length(tuple_cols))
        subheaders[perc_cols] .=  fill(LatexCell.(latexify("%")), length(perc_cols))
        subheaders[end] = LatexCell(L" ")
    else 
        subheaders = fill("", ncol(select_df)) 
        subheaders[tuple_cols] .=  "(ΛO, ΛΛ', Ψ)" 
        subheaders[perc_cols] .=  "%" 
        subheaders[end] = ""
    end 

    kwargs = (header = (names(select_df), subheaders),
            highlighters = (smallest_val_highlighter, smallest_val_highlighter2, largest_val_highlighter, min_PU_highlighter, min_PU_highlighter2), 
            formatters = round_formatter)
    if latex 
        kwargs = (header = (names(select_df), subheaders),
            highlighters = (latex_smallest_val_highlighter, latex_smallest_val_highlighter2, latex_largest_val_highlighter, latex_min_PU_highlighter, latex_min_PU_highlighter2), 
            formatters = round_formatter, 
            backend = Val(:latex))
    end 
    return PrettyDataFrame(select_df, (), kwargs)
end 

function show_mins(fmsimul, min_var, grouping = (); 
    summary_funs = (Bias = compute_bias, 
        Abs_bias = compute_abs_bias, 
        RMSE = compute_RMSE, 
        Var = compute_var, 
        PU = compute_PU),
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-6, 
    first_el = true)

    outdf, hlines = simul_summary(fmsimul.results , fmsimul.models, summary_funs; 
    excl_heywood = excl_heywood, 
    excl_no_conv = excl_no_conv, 
    excl_grad_nz = excl_grad_nz, 
    grad_cutoff = grad_cutoff, 
    print_only = false
    )

    #outdf = fmsimul.summary 

    forward_fill!(outdf.n)
    forward_fill!(outdf.Model)
    replace_empty_with_missing!(outdf)
    outdf = dropmissing(outdf) 

    tuple_cols = findall(values(mapcols(v -> all(isa.(v, Union{String, Tuple})), outdf)[1,:])) 
    perc_cols = findall(occursin.(r"%", names(outdf))) 
    subheaders = fill("", ncol(outdf)) 
    subheaders[tuple_cols] .=  "(ΛO, ΛΛ', Ψ)" 
    subheaders[perc_cols] .=  "%" 
    subheaders[length(subheaders)] = ""
    kwargs = (header = (names(outdf), subheaders), formatters = round_formatter,  highlighters = (smallest_val_highlighter, smallest_val_highlighter2, largest_val_highlighter, min_PU_highlighter, min_PU_highlighter2))
    PrettyDataFrame(combine(sdf -> filter(row -> filter_fun(row[min_var], sdf, min_var, first_el), sdf), groupby(outdf[3:end,:], grouping)), (), kwargs)
end 

function varimax(A; gamma = 1.0, minit = 20, maxit = 1000, reltol = 1e-12)
	d, m = size(A)

	if m == 1
		return A
	end

	if d == m && rank(A) == d
		return Matrix{Float64}(I, d, m)
	end

	T = Matrix{Float64}(I, m, m)
	B = A * T
    C = A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:]))
    if any(isnan.(C)) 
        return fill(NaN, d, m) 
    end 
    L,_,M = svd(C)  
	T = L * M'
	if norm(T-Matrix{Float64}(I, m, m)) < reltol
		T = qr(randn(m,m)).Q
		B = A * T
	end

	D = 0
	for k in 1:maxit
		Dold = D
		C = A' * (d*B.^3 - gamma*B * Diagonal(sum(B.^2, dims = 1)[:]))
        L,s,M = svd(C)  
		T = L * M'
		D = sum(s)
		B = A * T
		if (abs(D - Dold)/D < reltol) && k >= minit
			break
		end
	end

	for i in 1:m
		if abs(maximum(B[:,i])) < abs(minimum(B[:,i]))
			B[:,i] .= - B[:,i]
		end
	end

	return B
end


function filter_fun1(x, sdf, min_var)
    min_value = minimum(skipnan(abs.(first_element.(sdf[:, min_var]))))
    return abs(first_element(x)) == min_value
end

function filter_fun2(x, sdf, min_var)
    min_value = minimum(skipnan(abs.(second_element.(sdf[:, min_var]))))
    return abs(second_element(x)) == min_value
end

function filter_fun_PU1(x, sdf)
    min_value = abs(minimum(skipnan(abs.(first_element.(sdf[:,:PU])))) - .5 )
    return abs(first_element(x) - 0.5) == min_value
end

function filter_fun_PU2(x, sdf)
    min_value = abs(minimum(skipnan(abs.(second_element.(sdf[:,:PU])))) - .5 )
    return abs(second_element(x) - 0.5) == min_value
end

function filter_fun(x, sdf, min_var, first_el) 
    if min_var == :PU 
        if first_el 
            filter_fun_PU1(x, sdf)
        else
            filter_fun_PU2(x, sdf)
        end 
    else
        if first_el 
            filter_fun1(x, sdf, min_var)
        else 
            filter_fun2(x, sdf, min_var)
        end 
    end
end 

end 