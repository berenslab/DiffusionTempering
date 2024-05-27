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
@parameters α β γ δ
θ = [α => 1.5, β => 1.0, γ => 3.0, δ => 1.0]
prob_prior, prob = get_LV(θ)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [1 0]
data = generate_data(prob, proj, σ², dt_data)

loss = get_fenrir_loss()
optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())

# set loss parameters
logκ²0 = log(1e20)
diff_prior = Uniform(log.(1e-20), log.(1e50))
prior = merge_priors([prob_prior, diff_prior])
other_params = (data=data, prob=prob, prior=prior, σ²=σ², proj=proj, order=3, dt=dt_sol)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prior)
opt_bounds = forward.(get_prior_bounds(prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=false)

# run optimization loop
# --------------------------------------------
ps₀ = rand(prior, 100)'
ps₀[:, end] .= logκ²0 # set initial logκ²
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    sample_idx = lpad(idx, 3, "0")
    fout = "$out_dir" * "$exp_name-$sample_idx.csv"
    optprob = get_optprob(opt_loss, forward(p₀), other_params, opt_bounds)
    optsol = solve(optprob, optimizer, callback=callback(fout))
end
