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
@parameters l
θ = [l => 3]
prob_prior, prob = get_Pendulum(θ)

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
other_params =
    (data=data, prob=prob, σ²=σ², proj=proj, order=3, dt=dt_sol, prior=prob_prior)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prob_prior)
opt_bounds = forward.(get_prior_bounds(prob_prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=true)

# set tempering schedule
κ²0 = 20.0
# τ = 5.0; T(t::Real)::Float64 = 10.0^(κ²0 * exp(-t / τ))
T(t::Real)::Float64 = 10.0^(κ²0 - t)
tempering_schedule = T.(LinRange(0, 20, 21))

# run optimization loop
# --------------------------------------------
ps₀ = rand(prob_prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (iₛ, κ²ᵢ) in enumerate(tempering_schedule)
    for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
        sample_idx = lpad(idx, 3, "0")
        fout = "$out_dir" * "$exp_name-$iₛ-$sample_idx.csv"
        optprob = get_optprob(
            opt_loss,
            forward(p₀),
            (other_params..., logκ²=log.(κ²ᵢ)),
            opt_bounds,
        )
        optsol = solve(optprob, optimizer, callback=callback(fout, 0, time(), log.(κ²ᵢ)))
    end
end
