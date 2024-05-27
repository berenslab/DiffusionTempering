using DrWatson
@quickactivate "FenrirForNeuro"
using FenrirForNeuro
using LinearAlgebra, Statistics
using ModelingToolkit
using Optim, OptimizationOptimJL

using Random

# set up experiment
# --------------------------------------------
exp_name = splitext(basename(@__FILE__))[1]
out_dir = mkoutdir(exp_name; base_dir="./")

# set random seed
Random.seed!(1234)

# construct IVP
# --------------------------------------------
@parameters gNa gK
θ = [gNa => 25, gK => 7]
prob_prior, prob = get_SinglecompartmentHH(θ)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [1 0 0 0]
data = generate_data(prob, proj, σ², dt_data)

loss = get_fenrir_loss()
optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())

# set loss parameters
other_params = (
    data=data,
    prob=prob,
    σ²=σ²,
    proj=proj,
    order=3,
    dt=dt_sol,
    logκ²=log(1),
    prior=prob_prior,
)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prob_prior)
opt_bounds = forward.(get_prior_bounds(prob_prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=true)

# run optimization loop
# --------------------------------------------
ps₀ = rand(prob_prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    sample_idx = lpad(idx, 3, "0")
    fout = "$out_dir" * "$exp_name-$sample_idx.csv"
    optprob = get_optprob(opt_loss, forward(p₀), other_params, opt_bounds)
    optsol =
        solve(optprob, optimizer, callback=callback(fout, 0, time(), other_params.logκ²))
end
