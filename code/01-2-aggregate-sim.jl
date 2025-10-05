using JLD2, DataFrames, Printf, FiniteDiff, PrettyTables, LaTeXStrings

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")

include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
using .FactorModels
include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
using .FactorModelSimul

datasets = filter(x -> occursin("01-1", x), readdir(results_path))

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
fits = results.FMfit
no_Hess = map(x -> any(isnan.(x.hessian_pen)), fits)
no_grad = map(x -> any(isnan.(x.grad_pen)), fits)
fun(x) =  Main.FactorModels.pen_loglikl_full(x, S, p, q, n; 
    comp_pen = comp_pen, 
    psi_pen = psi_pen, 
    lambda_pen = lambda_pen, 
    scaling = scaling, 
    theta_fixed = theta_fixed
)
cache = FiniteDiff.HessianCache(fits[1].theta)
for i in findall(1 .- (1 .- no_Hess) .* (1 .- no_grad) .== 1)
    fit = fits[i]
    theta = fit.theta
    S = Main.FactorModels.get_covariance_mat(fit.X)
    p, q = size(fit.Lambda) 
    n = size(fit.X, 1) 
    comp_pen = fit.options.comp_pen
    psi_pen = fit.options.psi_pen
    lambda_pen = fit.options.lambda_pen
    scaling = fit.options.scaling
    theta_fixed = nothing
    if length(cache.xpp) != (p * (q + 1)) 
        cache = FiniteDiff.HessianCache(theta)
    end 
    if !any(isnan.(theta))
        fits[i].hessian_pen .= FiniteDiff.finite_difference_hessian(fun, theta, cache) 
        fits[i].grad_pen .= FiniteDiff.finite_difference_gradient(fun, theta, cache) 
    end
end

sumry, hlines, subheaders = Main.FactorModelSimul.simul_summary(results, models; 
    excl_heywood = true, 
    excl_no_conv = true, 
    excl_grad_nz = true, 
    grad_cutoff = 1e-4,  
    heywood_cutoff = 1e-4, 
    print_only = false, 
    model_select = false 
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
    false
) 

@save joinpath(results_path, "01-2-full-sim.jld2") sim_results
