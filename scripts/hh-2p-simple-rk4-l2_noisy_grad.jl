using DrWatson
@quickactivate "FenrirForNeuro"
using FenrirForNeuro
using LinearAlgebra, Statistics
using ModelingToolkit
using Optim, OptimizationOptimJL
using ForwardDiff

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
proj = [0 0 0 1]
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

function get_noisy_optprob(
    obj,
    p₀,
    other_params,
    bounds=nothing;
    abstol=1e-9,
    reltol=1e-6,
    noise_scale=0.1,
    decay_schedule=nothing,
)
    j = 0
    decay_schedule = isnothing(decay_schedule) ? (t -> 1.0) : decay_schedule
    function noisy_gradient(G, u, p)
        ForwardDiff.gradient!(G, u -> obj(u, p)[1], u)
        G .+= decay_schedule(j) * noise_scale * randn(length(u))
        j += 1
    end

    # Define the optimization function with the noisy gradient
    f = OptimizationFunction(obj, grad=noisy_gradient)
    lbounds, ubounds = isnothing(bounds) ? (nothing, nothing) : bounds
    optprob = OptimizationProblem(
        f,
        p₀,
        other_params;
        lb=lbounds,
        ub=ubounds,
        abstol=abstol,
        reltol=reltol,
    )
    return optprob
end

# run optimization loop
# --------------------------------------------
ps₀ = rand(prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
noise_scales = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
exp_decay(t, τ=5) = exp(-t / τ)
for (iₜ, decay_with) in enumerate([nothing, t -> exp_decay(t, 5), t -> exp_decay(t, 10)])
    for (iₛ, s) in enumerate(noise_scales)
        for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
            sample_idx = lpad(idx, 3, "0")
            fout = "$out_dir" * "$exp_name" * "_$iₜ" * "_$iₛ-$sample_idx.csv"
            optprob = get_noisy_optprob(
                opt_loss,
                forward(p₀),
                other_params,
                opt_bounds;
                noise_scale=s,
                decay_schedule=decay_with,
            )
            optsol = solve(optprob, optimizer, callback=callback(fout))
        end
    end
end