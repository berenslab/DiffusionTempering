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
@parameters gNa gK gleak
θ = [gNa => 25, gK => 7, gleak => 0.1]
prob_prior, prob = get_SinglecompartmentHH(θ)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [1 0 0 0]
data = generate_data(prob, proj, σ², dt_data)

loss = get_fenrir_loss() # or get_l2_loss()
optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())

# set loss parameters
diff_prior = Product(fill(Uniform(log.(1e-1), log.(1e21)), 1))
other_params = (data=data, prob=prob, σ²=σ², proj=proj, order=3, dt=dt_sol)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prob_prior)
opt_bounds = forward.(get_prior_bounds(prob_prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=true)

# find optimal diffusion for true parameters
# --------------------------------------------
diff_callback = function ()
    j = 0
    function (θ, l, ts, states)
        j += 1

        κ² = exp.(θ[1])
        @info "[Iteration $j] Callback" κ² l

        GC.gc() # fixes memory leak in ForwardDiff
        return false
    end
end

# run optimization loop
# --------------------------------------------
ps₀ = rand(prob_prior, 100)'
logκ²s₀ = rand(diff_prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (idx, p₀, logκ²₀) in zip(from:to, eachrow(ps₀[from:to, :]), eachrow(logκ²s₀[from:to]))
    optprob = get_optprob(
        loss,
        [logκ²₀...],
        (other_params..., θₚ=map(θᵢ -> θᵢ[2], θ), prior=diff_prior),
        get_prior_bounds(diff_prior),
    )
    optsol = solve(optprob, optimizer, callback=diff_callback())
    logκ²_opt = optsol.u[1]

    sample_idx = lpad(idx, 3, "0")
    fout = "$out_dir" * "$exp_name-$sample_idx.csv"
    optprob = get_optprob(
        opt_loss,
        forward(p₀),
        (other_params..., logκ²=logκ²_opt, prior=prob_prior),
        opt_bounds,
    )
    optsol = solve(optprob, optimizer, callback=callback(fout, 0, time(), logκ²_opt))
end
