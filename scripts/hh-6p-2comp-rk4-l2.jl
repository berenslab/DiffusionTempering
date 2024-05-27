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
@parameters gNa₁ gK₁ gleak₁ gNa₂ gK₂ gleak₂
θ = [gNa₁ => 25, gK₁ => 7, gleak₁ => 0.09, gNa₂ => 20, gK₂ => 10, gleak₂ => 0.11]

A = get_Asoma(15.3, 184) / 2
@named c₁ = SimpleHHSystem(; gNa=gNa₁, gK=gK₁, gleak=gleak₁, A=A)
@named c₂ = SimpleHHSystem(; gNa=gNa₂, gK=gK₂, gleak=gleak₂, Iₑ=I₀, A=A)
compartments = [c₁, c₂]
G = get_cable_matrix([1])

prob_prior, prob = get_MulticompartmentHH(θ, [c₁, c₂], G)
FIX_PRIOR(prior) = Product([v for v in prior.v[[2, 3, 1, 6, 5, 4]]]) # HOTFIX to align prior and model param order
prob_prior = FIX_PRIOR(prob_prior)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [
    1 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0
]
data = generate_data(prob, proj, σ², dt_data)

loss = get_l2_loss()
optimizer = LBFGS(linesearch=Optim.LineSearches.BackTracking())

# set loss parameters
prior = prob_prior
other_params = (data=data, prob=prob, proj=proj)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prior)
opt_bounds = forward.(get_prior_bounds(prior))

# define callback
callback = get_rk4_l2_callback(prob, backward)

# run optimization loop
# --------------------------------------------
ps₀ = rand(prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    sample_idx = lpad(idx, 3, "0")
    fout = "$out_dir" * "$exp_name-$sample_idx.csv"
    optprob = get_optprob(opt_loss, forward(p₀), other_params, opt_bounds)
    optsol = solve(optprob, optimizer, callback=callback(fout))
end