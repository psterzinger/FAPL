module FactorModels

using LinearAlgebra, Optim, Statistics, ForwardDiff, FiniteDiff

export fit_factor_model, psi_Huber, lambda_Huber, lambda_one_sided_pen, psi_stronger_pen, lambda_eigen_pen, lambda_cov_mat_pen, akaike_pen, hirose_pen, get_lambda, get_psi, FMfit

#Function to fit factor models 
function fit_factor_model_ML(X,q;  
    scaling = nothing, 
    psi_pen = nothing, 
    lambda_pen = nothing, 
    comp_pen = nothing, 
    theta_start = nothing, 
    theta_fixed = nothing, 
    optimizer = Optim.BFGS(),
    abstol = 1e-9, 
    reltol = 1e-9, 
    max_iter = 1000, 
    covar = false, 
    n = nothing)
    if covar 
        S = copy(X)
        p = size(X, 1)
    else 
        S = get_covariance_mat(X) 
        (n,p) = size(X)
    end 
    opt_res = Optim.optimize(x -> pen_loglikl_full(x, S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling, theta_fixed = theta_fixed), theta_start, method = optimizer, x_abstol = abstol, f_reltol = reltol, iterations = max_iter) 
    return Optim.minimizer(opt_res), Optim.converged(opt_res)
    end 

    function fit_factor_model_EM(X,q; scaling = nothing, 
            psi_pen = nothing, 
            lambda_pen = nothing, 
            comp_pen = nothing, 
            theta_start = nothing, 
            theta_fixed = nothing, 
            optimizer = Optim.Newton(), 
            max_iter = 1000, 
            abstol = 1e-9, 
            reltol = 1e-9,
            covar = false, 
            n = nothing)
    if covar 
        S = copy(X)
        p = size(X, 1)
    else 
        S = get_covariance_mat(X) 
        (n,p) = size(X)
    end  

    if isnothing(theta_fixed)
        theta_old = theta_start 
        start_val = similar(theta_old)
    else
        theta_old = make_full_theta(theta_start, theta_fixed) 
        start_val = similar(theta_start) 
    end 
    lambda_old = get_lambda(theta_old, p, q)
    psi_old = get_psi(theta_old, p, q) 
    theta_new = similar(theta_old) 

    sigma_invl = inv(lambda_old * lambda_old' + diagm(psi_old)) * lambda_old
    Sxy = S * sigma_invl
    Syy = inv(I(q) + lambda_old' * diagm(1.0 ./ psi_old) * lambda_old) + sigma_invl' * S * sigma_invl

    conv = false 
    count = 1 
    while !conv && count <= max_iter 
        if isnothing(theta_fixed)
            start_val .= theta_old 
        else
            start_val .= theta_old[isnan.(theta_fixed)] 
        end 
        opt_res = Optim.optimize(x ->  cond_pen_loglikl_full(x, S, Sxy, Syy, n, p, q; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling, theta_fixed = theta_fixed), start_val, method = optimizer) 
        if isnothing(theta_fixed)
            theta_new .= Optim.minimizer(opt_res) 
        else
            theta_new .= make_full_theta(Optim.minimizer(opt_res), theta_fixed)
        end 

        conv = maximum(abs.(theta_new - theta_old) ./ abs.(theta_new)) < reltol || maximum(abs.(theta_new - theta_old)) < abstol 
        theta_old .= theta_new 

        lambda_old .= get_lambda(theta_old, p, q) 
        psi_old .= get_psi(theta_old, p, q) 
        sigma_invl .=  inv(lambda_old * lambda_old' + diagm(psi_old)) * lambda_old
        Sxy .= S * sigma_invl 
        Syy .= inv(I(q) + lambda_old' * diagm(1. ./ psi_old) * lambda_old) + sigma_invl' * S * sigma_invl

        count += 1 
    end 
    return theta_new, conv 
end 

function fit_factor_model_EM_nopen(X,q; 
    theta_start = nothing, 
    max_iter = 1000, 
    abstol = 1e-9, 
    reltol = 1e-9,
    covar = false, 
    n = nothing) 
    
    if covar 
        S = copy(X) 
        p = size(X, 1) 
    else
        n,p = size(X)
        S = get_covariance_mat(X) 
    end 

    theta_old = theta_start 
    lambda_old = get_lambda(theta_old, p, q)
    psi_old = get_psi(theta_old, p, q) 
    theta_new = similar(theta_old) 
    psi_new = similar(psi_old) 
    lambda_new = similar(lambda_old)

    sigma_invl = inv(lambda_old * lambda_old' + diagm(psi_old)) * lambda_old
    Sxy = S * sigma_invl
    Syy = inv(I(q) + lambda_old' * diagm(1.0 ./ psi_old) * lambda_old) + sigma_invl' * S * sigma_invl

    conv = false 
    count = 1 
    while !conv && count <= max_iter  
        lambda_new .= Sxy * inv(Syy) 
        psi_new .= diag(S - 2.0 * Sxy * lambda_old' + lambda_old * Syy * lambda_old')
        theta_new .= vcat(vec(lambda_new), psi_new) 

        conv = maximum(abs.(theta_new - theta_old) ./ abs.(theta_new)) < reltol || maximum(abs.(theta_new - theta_old)) < abstol 
        
        theta_old .= theta_new 
        lambda_old .= lambda_new 
        psi_old .= psi_new 

        sigma_invl = inv(lambda_old * lambda_old' + diagm(psi_old)) * lambda_old
        Sxy = S * sigma_invl
        Syy = inv(I(q) + lambda_old' * diagm(1.0 ./ psi_old) * lambda_old) + sigma_invl' * S * sigma_invl

        count +=1
    end 
    return theta_new, conv 
end 

function fit_factor_model(X,q;  
    scaling = nothing, 
    psi_pen = nothing, 
    lambda_pen = nothing, 
    comp_pen = nothing, 
    theta_start = nothing, 
    theta_fixed = nothing,
    method = :ML, 
    optimizer = Optim.BFGS(), 
    max_iter = 1000, 
    abstol = 1e-9, 
    reltol = 1e-9, 
    raw = false, 
    max_iter_EM = 100, 
    factor_method = :Bartlett,
    covar = false,
    n = nothing) 

    @assert typeof(X) == Matrix{Float64}
    if covar 
        p = size(X, 1) 
        S = copy(X) 
        @assert !isnothing(n) 
    else 
        n, p = size(X)
        @assert p > q 
        S = get_covariance_mat(X) 
    end     
    @assert rank(S) == p 

    if !isnothing(scaling) 
        if isa(scaling, Function) 
            scaling = scaling(n, p, q) 
            if length(scaling) == 1 
                scaling = fill(scaling, 2) 
            end 
        end 
        @assert length(scaling) ∈ (1,2)
        @assert all(isreal.(scaling))
        if length(scaling) == 1
            scaling = fill(scaling[1],2)
        end 
    else
        scaling = 0.5 * [1, 1 / q] / p#fill(0.5 / (log(det(S)) + p),2)
    end

    if !isnothing(theta_start) 
        if isnothing(theta_fixed)
            @assert length(theta_start) == (q + 1) * p 
        else
            @assert length(theta_start) == sum(isnan.(theta_fixed)) 
        end 
    else 
        if isnothing(theta_fixed)
            theta_start = vcat(lambda_start(S, q), fill(1.0, p)) 
        else 
            theta_start = vcat(lambda_start(S, q), fill(1.0, p))[isnan.(theta_fixed)]
        end      
    end 

    if method == :EM || method == :both 
        if method == :both 
            iter = max_iter_EM 
        else 
            iter = max_iter 
        end 
        if isnothing(theta_fixed) && isnothing(psi_pen) && isnothing(lambda_pen) && isnothing(comp_pen)
            theta, conv = fit_factor_model_EM_nopen(X,q; theta_start = theta_start, max_iter = iter, abstol = abstol, reltol = reltol, covar = covar, n = n)
        else
            theta, conv = fit_factor_model_EM(X,q; theta_start = theta_start, max_iter = iter, abstol = abstol, reltol = reltol, scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, optimizer = optimizer, theta_fixed = theta_fixed, covar = covar, n = n)
        end 
    end 
    if method == :ML 
        theta, conv = fit_factor_model_ML(X,q; theta_start = theta_start, max_iter = max_iter, abstol = abstol, reltol = reltol, scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, optimizer = optimizer, theta_fixed = theta_fixed, covar = covar, n = n)
        if !isnothing(theta_fixed)
            theta = make_full_theta(theta, theta_fixed)
        end 
    elseif method == :both 
        theta, conv = fit_factor_model_ML(X,q; theta_start = theta, max_iter = max_iter, abstol = abstol, reltol = reltol, scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, optimizer = optimizer, theta_fixed = theta_fixed, covar = covar, n = n)
        if !isnothing(theta_fixed)
            theta = make_full_theta(theta, theta_fixed)
        end 
    elseif method != :EM
        @warn "Maximisation method must be either :ML, :EM or :both, continuing with :ML"
        theta, conv = fit_factor_model_ML(X,q; theta_start = theta_start, max_iter = max_iter, abstol = abstol, reltol = reltol, scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, optimizer = optimizer, theta_fixed = theta_fixed, covar = covar, n = n)
        if !isnothing(theta_fixed)
            theta = make_full_theta(theta, theta_fixed)
        end 
    end
    if raw 
        return theta, conv 
    else 
        models = nothing 
        options = (scaling = scaling, psi_pen = psi_pen, lambda_pen = lambda_pen, comp_pen = comp_pen, method = method, optimizer = optimizer, max_iter = max_iter, abstol = abstol, reltol = reltol, covar = covar, n = n)
        Lambda = get_lambda(theta, p, q)
        Psi = Diagonal(get_psi(theta, p, q)) 
        Sigma = get_sigma(theta, p, q) 
        loglik = -0.5 * n * loglikl(Sigma, S)
        loglik_pen = pen_loglikl_full(theta, S, p, q, n; psi_pen = psi_pen, lambda_pen = lambda_pen, comp_pen = comp_pen, scaling = scaling)
        grad_pen = fill(NaN, length(theta))
        hessian_pen  = fill(NaN, (length(theta), length(theta))) 
        try
            grad_pen = ForwardDiff.gradient(x -> pen_loglikl_full(x, S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling, theta_fixed = theta_fixed), theta) 
            hessian_pen = ForwardDiff.hessian(x -> pen_loglikl_full(x, S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling, theta_fixed = theta_fixed), theta)
        catch 
            fun(x) =  pen_loglikl_full(x, S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling, theta_fixed = theta_fixed)
            grad_pen =  FiniteDiff.finite_difference_gradient(fun, theta)
            hessian_pen = FiniteDiff.finite_difference_hessian(fun, theta)  
        end 
        heywood = any(diag(Psi) .< 1e-5) || any(diag(Psi) .> 1e5) || any(diag(Lambda * Lambda') .> 1e5) # || any(diag(Lambda * Lambda') .< 1e-5)
        factor_scores = compute_factor_scores(X, Lambda, Psi; method = factor_method) 
        return FMfit(X, theta, Lambda, Psi, Sigma, factor_scores, conv, loglik, loglik_pen, grad_pen, hessian_pen, q, heywood, models, options)     
    end 
end 

function fit_factor_model(X;  
    scaling = nothing, 
    psi_pen = nothing, 
    lambda_pen = nothing, 
    comp_pen = nothing,
    method = :ML, 
    optimizer = Optim.BFGS(), 
    max_iter = 1000, 
    max_iter_EM = 100, 
    abstol = 1e-9, 
    reltol = 1e-9, 
    model_select = :AIC, 
    factor_method = :Bartlett, 
    covar = false, 
    n = nothing)  

    @assert model_select ∈ (:AIC, :BIC)
    if covar 
        S = copy(X) 
        p = size(X, 1) 
    else 
        (n, p) = size(X)
        S = get_covariance_mat(X)
    end 
    AIC = fill(Inf, p - 1)
    BIC = fill(Inf, p - 1)
    dfs = fill(NaN, p - 1)
    thetas = [Float64[] for _ in 1:(p - 1)]
    convs = Vector{Bool}(undef,p - 1)
    for q in 1:(p - 1) 
        df = p * (p - 1) / 2 - p * q + q * (q - 1) / 2 
        dfs[q] = df 
        if df >= 0
            theta, conv = fit_factor_model(X, q; scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, theta_start = nothing, method = method, optimizer = optimizer, max_iter = max_iter, abstol = abstol, reltol = reltol, max_iter_EM = max_iter_EM, raw = true, covar = covar, n = n) 
            convs[q] = conv
            thetas[q] = theta 
            Lambda = get_lambda(theta, p, q) 
            Psi = get_psi(theta, p, q) 
            if !conv || any(Psi .< 1e-5) || any(Psi .> 1e5) || any(diag(Lambda * Lambda') .> 1e5) 
                AIC[q] = Inf 
                BIC[q] = Inf 
            else model_select
                AIC[q] = n * loglikl(get_sigma(theta, p, q), S) + 2 * p * (q + 1) - q * (q - 1)
                BIC[q] = n * loglikl(get_sigma(theta, p, q), S) + (p * (q + 1) - q * (q - 1) / 2)  * log(n)
            end 
        else 
            break 
        end 
    end 
    models = (thetas = thetas, AIC = AIC, BIC = BIC, dfs = dfs, convs = convs, qs = 1:(p - 1))
    options = (scaling = scaling, comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, method = method, optimizer = optimizer, max_iter = max_iter, abstol = abstol, reltol = reltol, model_select = model_select, max_iter_EM = max_iter_EM, factor_method = factor_method, covar = covar, n = n)
    q_AIC = argmin(AIC) 
    theta_AIC = thetas[q_AIC] 
    lambda_AIC = get_lambda(theta_AIC, p, q_AIC)
    psi_AIC = Diagonal(get_psi(theta_AIC, p, q_AIC))
    sigma_AIC = get_sigma(theta_AIC, p, q_AIC)
    q_BIC = argmin(BIC) 
    theta_BIC = thetas[q_BIC] 
    lambda_BIC = get_lambda(theta_BIC, p, q_BIC)
    psi_BIC = Diagonal(get_psi(theta_BIC, p, q_BIC))
    sigma_BIC = get_sigma(theta_BIC, p, q_BIC)
    q = (q_AIC, q_BIC)
    theta = (theta_AIC, theta_BIC) 
    conv = (convs[q_AIC], convs[q_BIC])
    Lambda = (lambda_AIC, lambda_BIC)
    Psi = (psi_AIC, psi_BIC)
    Sigma = (sigma_AIC, sigma_BIC) 
    loglik = (-0.5 * n * loglikl(sigma_AIC, S),  -0.5 * n * loglikl(sigma_BIC, S))
    hessian_pen = nothing 
    grad_pen = nothing 
    if isa(scaling, Function) 
        scale_AIC = scaling(n, p, q_AIC) 
        if length(scale_AIC) == 1 
            scale_AIC = fill(scale_AIC, 2) 
        end 
        scale_BIC = scaling(n, p, q_BIC) 
        if length(scale_BIC) == 1 
            scale_BIC = fill(scale_BIC, 2) 
        end 
        loglik_pen = (pen_loglikl_full(theta_AIC, S, p, q_AIC, n; psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_AIC), pen_loglikl_full(theta_BIC, S, p, q_BIC, n; psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_BIC))
        try
            grad_pen = (ForwardDiff.gradient(x -> pen_loglikl_full(x, S, p, q_AIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_AIC), thetas[q_AIC]),
                        ForwardDiff.gradient(x -> pen_loglikl_full(x, S, p, q_BIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_BIC), thetas[q_BIC])) 
            hessian_pen = (ForwardDiff.hessian(x -> pen_loglikl_full(x, S, p, q_AIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_AIC), thetas[q_AIC]), 
                            ForwardDiff.hessian(x -> pen_loglikl_full(x, S, p, q_BIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_BIC), thetas[q_BIC]))
        catch 
            funA(x) =  pen_loglikl_fullpen_loglikl_full(x, S, p, q_AIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_AIC)
            funB(x) =  pen_loglikl_fullpen_loglikl_full(x, S, p, q_BIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scale_BIC)
            grad_pen =  (FiniteDiff.finite_difference_gradient(funA, thetas[q_AIC]), FiniteDiff.finite_difference_gradient(funB, thetas[q_BIC]))
            hessian_pen = (FiniteDiff.finite_difference_hessian(funA, thetas[q_AIC]), FiniteDiff.finite_difference_hessian(funB, thetas[q_BIC]))
        end 
    else 
        loglik_pen = (pen_loglikl_full(thetas[q_AIC], S, p, q_AIC, n; psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling), pen_loglikl_full(thetas[q_BIC], S, p, q_BIC, n; psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling))
        try
            grad_pen = (ForwardDiff.gradient(x -> pen_loglikl_full(x, S, p, q_AIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling), thetas[q_AIC]),
                        ForwardDiff.gradient(x -> pen_loglikl_full(x, S, p, q_BIC, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling), thetas[q_BIC])) 
            hessian_pen = (ForwardDiff.hessian(x -> pen_loglikl_full(x, S, p, q_AIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling), thetas[q_AIC]), 
                            ForwardDiff.hessian(x -> pen_loglikl_full(x, S, p, q_BIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling), thetas[q_BIC]))
        catch 
            funA(x) =  pen_loglikl_fullpen_loglikl_full(x, S, p, q_AIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling)
            funB(x) =  pen_loglikl_fullpen_loglikl_full(x, S, p, q_BIC, n;comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling)
            grad_pen =  (FiniteDiff.finite_difference_gradient(funA, thetas[q_AIC]), FiniteDiff.finite_difference_gradient(funB, thetas[q_BIC]))
            hessian_pen = (FiniteDiff.finite_difference_hessian(funA, thetas[q_AIC]), FiniteDiff.finite_difference_hessian(funB, thetas[q_BIC]))
        end 
    end 
    heywood = (any(diag(psi_AIC) .< 1e-5) || any(diag(psi_AIC) .> 1e5) || any(diag(lambda_AIC * lambda_AIC') .> 1e5), 
                any(diag(psi_BIC) .< 1e-5) || any(diag(psi_BIC) .> 1e5) || any(diag(lambda_BIC * lambda_BIC') .> 1e5))
    factor_scores = () #(compute_factor_scores(X, lambda_AIC, psi_AIC; method = factor_method), compute_factor_scores(X, lambda_BIC, psi_BIC; method = factor_method))
    return FMfit(X, theta, Lambda, Psi, Sigma, factor_scores, conv, loglik, loglik_pen, grad_pen, hessian_pen, q, heywood, models, options) 
end 

#likelihoods
function loglikl(Sigma,S)  
    try
        logdet(Sigma) + tr(inv(Sigma) * S)
    catch 
        Inf
    end 
end 

function pen_loglikl_full(theta, S, p, q, n; comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, scaling = nothing, theta_fixed = nothing) 
    if !isnothing(theta_fixed) 
        pen_loglikl(make_full_theta(theta, theta_fixed), S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling)
    else 
        pen_loglikl(theta, S, p, q, n; comp_pen = comp_pen, psi_pen = psi_pen, lambda_pen = lambda_pen, scaling = scaling)
    end 
end 

function pen_loglikl(theta, S, p, q, n; comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, scaling = nothing)
    psi = get_psi(theta, p, q) 
    lambda = get_lambda(theta, p, q)
    val = loglikl(lambda * lambda' + diagm(psi), S)
    if isnothing(comp_pen)
        if !isnothing(psi_pen)
            val += 2.0 * scaling[1] * psi_pen(psi) / n 
        end
        if !isnothing(lambda_pen)
            val += 2.0 * scaling[2] * lambda_pen(lambda) / n 
        end
    else
        val += 2.0 * scaling[1] * comp_pen(lambda, psi, S) / n 
    end 
    return val
end

function cond_loglikl(psi, lambda, S, Sxy, Syy)
    if any(psi .<= sqrt(eps()))
        return Inf
    else
        return sum(log.(psi)) + tr((S + (lambda * Syy - 2.0 * Sxy) * lambda') ./ psi)
    end 
end 

function cond_pen_loglikl(theta, S, Sxy, Syy, n, p, q; scaling = nothing, comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing) 
    psi = get_psi(theta, p, q) 
    lambda = get_lambda(theta, p, q) 
    val = cond_loglikl(psi, lambda, S, Sxy, Syy) 
    
    if isnothing(comp_pen) 
        if !isnothing(psi_pen)
            val += 2.0 * scaling[1] * psi_pen(psi) / n 
        end 
        if !isnothing(lambda_pen) 
            val += 2.0 * scaling[2] * lambda_pen(lambda) / n 
        end 
    else 
        val += 2.0 * scaling[1] * comp_pen(lambda, psi, S) / n 
    end 
    return val 
end 

function cond_pen_loglikl_full(theta, S, Sxy, Syy, n, p, q; scaling = nothing, comp_pen = nothing, psi_pen = nothing, lambda_pen = nothing, theta_fixed = nothing)
    if !isnothing(theta_fixed) 
        cond_pen_loglikl(make_full_theta(theta, theta_fixed), S, Sxy, Syy, n, p, q; comp_pen = comp_pen, scaling = scaling, psi_pen = psi_pen, lambda_pen = lambda_pen)
    else 
        cond_pen_loglikl(theta, S, Sxy, Syy, n, p, q; comp_pen = comp_pen, scaling = scaling, psi_pen = psi_pen, lambda_pen = lambda_pen)
    end 
end

#penalties 
function Huber_loss(x; delta = 1) 
    ax = abs(x)
    if ax <= 1
        0.5 * x^2 
    else
        delta*(ax - 0.5 * delta)
    end 
end 

function one_side_pen(x; delta = 1) 
    if x <= 1
        (1 + delta) * (x^(1 / (1 + delta))) 
    else
        log(x) + (1 + delta)
    end 
end 

function stronger_psi(x; delta = 1) 
    if x < 0 
        Inf
    elseif x <= 1 
        x^-(1 + delta) + (2 + delta) * x - 2
    else
        log(x) + (1 + delta) 
    end 
end 

function psi_Huber(psi)
    try 
        sum(Huber_loss.(log.(psi))) 
    catch 
        Inf 
    end
end 

function psi_stronger_pen(psi)
    try 
        sum(stronger_psi.(psi)) 
    catch 
        Inf 
    end
end 

function lambda_Huber(lambda) 
    try 
        # sum(Huber_loss.(log.(sum(lambda.^2, dims = 2))))
        sum(Huber_loss.(sum(lambda.^2, dims = 2)))
    catch 
        Inf
    end 
end 

function lambda_one_sided_pen(lambda) 
    try 
        sum(one_side_pen.(sum(lambda.^2, dims = 2)))
    catch 
        Inf
    end 
end 

function lambda_eigen_pen(lambda)
    p, q = size(lambda) 
    e, v = eigen(lambda * lambda') 
    if any(e[1:p - q] .> sqrt(eps())) || any(e[p - q + 1:end] .< sqrt(eps())) 
        Inf 
    else
        sum(Huber_loss.(e[p - q + 1:end])) 
    end 
end 

function lambda_cov_mat_pen(lambda, S)
    q = size(lambda, 2) 

    e,v = eigen(S) 
    sinds = arg_n_largest_values(e, q)
   
    el, vl = eigen(lambda * lambda') 
    linds = arg_n_largest_values(el, q)
    
    sum(Huber_loss.((e[sinds] - el[linds]))) + sum(Huber_loss.(vec(v[:,sinds] - vl[:,linds])))
end

function akaike_pen(lambda, psi, S)
    tr(lambda * lambda' * inv(Diagonal(psi))) 
end 

function hirose_pen(lambda, psi, S) 
    tr(S * inv(Diagonal(psi))) 
end 

#parameter transformation
function make_full_theta(theta, theta_fixed) 
    theta_full = similar(theta_fixed)
    theta_full[isnan.(theta_fixed)] .= theta 
    theta_full[.!isnan.(theta_fixed)] .= theta_fixed[.!isnan.(theta_fixed)]
    return theta_full
end 

function get_psi(theta, p, q) 
    theta[p * q + 1 : end] 
end 

function get_lambda(theta, p, q) 
    reshape(theta[1 : p * q], p, q)
end 

function get_sigma(theta, p, q) 
    lambda = get_lambda(theta, p ,q) 
    psi = get_psi(theta, p, q)
    lambda * lambda' + diagm(psi) 
end 

#starting values 
function psi_start(S, q) 
    lambda = reshape(lambda_start(S, q), size(S,1), q)  
    diag((S - lambda * lambda').^2) / 2 
end 

function lambda_start(S, q) 
    e,v = eigen(S) 
    sinds = arg_n_largest_values(e, q)
    vec(v[:,sinds] * diagm(sqrt.(e[sinds])))
end 

#helpers misc 
struct FMfit 
    X::Matrix{Float64} 
    theta::Union{Vector, Tuple}
    Lambda::Union{Matrix, Tuple}
    Psi::Union{Diagonal{Float64, Vector{Float64}}, Tuple}
    Sigma::Union{Matrix, Tuple}
    factor_scores::Union{Matrix{Float64}, Tuple}
    conv::Union{Bool, Float64, Tuple}
    loglikl::Union{Float64, Tuple}
    loglikl_pen::Union{Float64, Tuple}
    grad_pen::Union{Vector, Tuple}
    hessian_pen::Union{Matrix, Tuple}
    q::Union{Int64, Tuple}
    Heywood::Union{Bool, Float64, Tuple}
    models::Union{Nothing, NamedTuple} 
    options::Any
end 

function compute_factor_scores(X, Lambda, Psi; method = :Bartlett)
    @assert method ∈ [:Bartlett, :ML]
    if method == :ML 
        mat = Lambda' * inv(Lambda * Lambda' + I(size(Lambda, 1))) 
    else
        mat = inv(Lambda' * inv(Psi) * Lambda) * Lambda' * inv(Psi)  
    end 
    try 
        return (X  .- mean(X; dims = 1)) * mat'
    catch e
        @error "L:$Lambda, P:$Psi, mat:$mat, mu:$mu, X:$X" exception=(e, catch_backtrace())
        rethrow(e)
    end 
end 


function min_orth_rotation(L, target) 
    SDU = svd(target' * L) 
    return SDU.U * SDU.Vt
end

function get_covariance_mat(X) 
    means = mean(X; dims = 1)
    (X'X) / size(X, 1) - means' * means  
end 

function arg_n_largest_values(A::AbstractArray{T,N}, n::Integer) where {T,N}
    perm = sortperm(-vec(A))
    ci = CartesianIndices(A)
    return ci[perm[1:n]]
end

function arg_n_smallest_values(A::AbstractArray{T,N}, n::Integer) where {T,N}
    perm = sortperm(vec(A))
    ci = CartesianIndices(A)
    return ci[perm[1:n]]
end

end 