using DataFrames, Distributed, LinearAlgebra, JLD2 

num_workers = 8 
addprocs(num_workers; topology=:master_worker)

@everywhere begin 
    using Random, Distributions, LinearAlgebra, Optim

    supp_path = abspath(joinpath(@__DIR__, ".."))
    results_path = joinpath(supp_path, "results")
    figures_path = joinpath(supp_path, "figures")

    function soft_scale(n, p, q) 
        sqrt(2 / n) 
    end 
    function const_scale(n, p, q) 
        0.5  * n 
    end 
end 
@everywhere include(joinpath(supp_path, "code", "methods", "FactorModels.jl"))
@everywhere using .FactorModels
@everywhere include(joinpath(supp_path, "code", "methods", "FactorModelSimul.jl"))
@everywhere using .FactorModelSimul

lambda_1 = kron(I(3), [.8, .65, .4])
psi_1 = Diagonal(diag(I(9) - lambda_1 * lambda_1'))

lambda_2 = kron(I(3), [.8, .65, .5, .35, .2])
psi_2 = Diagonal(diag(I(15) - lambda_2 * lambda_2'))

lambda_3 = kron(I(3), 0.8:-0.1:0.1) 
psi_3 = Diagonal(diag(I(24) - lambda_3 * lambda_3'))

lambda_4 = kron(I(3), fill(.8,3))
ind = lambda_4[:,3] .!= 0.0
lambda_4[ind,3] .= .3
psi_4 = Diagonal(diag(I(9) - lambda_4 * lambda_4'))

lambda_5 = kron(I(3), fill(.8,5))
ind = lambda_5[:,3] .!= 0.0
lambda_5[ind,3] .= .3
psi_5 = Diagonal(diag(I(15) - lambda_5 * lambda_5'))

lambda_6 = kron(I(3), fill(.8,8))
ind = lambda_6[:,3] .!= 0.0
lambda_6[ind,3] .= .3
psi_6 = Diagonal(diag(I(24) - lambda_6 * lambda_6'))

models = (A_3 = (lambda_1, psi_1), B_3 = (lambda_4, psi_4))   
scalings = (:const_scale, :soft_scale) 
penalties = ((:nothing, :nothing, :akaike_pen), (:nothing, :nothing, :hirose_pen)) 

narray = [50, 400, 1000] 
reps = 1000

model_keys = collect(keys(models)) 
combinations = [(model, n) for model in model_keys for n in narray]

for combination in combinations
    model_key = Symbol(combination[1]) 
    model_value = models[model_key] 
    selected_model = NamedTuple{Tuple([model_key])}(Tuple([model_value]))
    selected_n = combination[2] 

    full_sim = Main.FactorModelSimul.factor_model_simul(
        selected_model,
        penalties, 
        scalings, 
        [selected_n],
        reps;  
        optimizer = Optim.Newton(),
        method = :both, 
        max_iter_EM = 100, 
        model_select = true 
    )

    filename = "03-1-fapl-MS-sim-$(model_key)-n$(selected_n).jld2"
    @save joinpath(results_path, filename) full_sim 
end 