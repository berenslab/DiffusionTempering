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
logκ²0 = log(1e20)
diff_prior = Uniform(log.(1e-20), log.(1e50))
prior = merge_priors([prob_prior, diff_prior])
other_params = (data=data, prob=prob, prior=prior, σ²=σ², proj=proj, order=3, dt=dt_sol)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prior)
opt_bounds = forward.(get_prior_bounds(prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=false)

function get_noisy_optprob(
    obj,
    p₀,
    other_params,
    bounds=nothing;
    abstol=1e-9,
    reltol=1e-6,
    noise_scale=0.1,
)
    function noisy_gradient(G, u, p)
        ForwardDiff.gradient!(G, u -> obj(u, p)[1], u)
        G .+= noise_scale * randn(length(u))
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
for (iₛ, s) in enumerate([0, 0.01, 0.1, 0.5, 1.0])
    for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
        sample_idx = lpad(idx, 3, "0")
        fout = "$out_dir" * "$exp_name" * "_$iₛ-$sample_idx.csv"
        optprob = get_noisy_optprob(
            opt_loss,
            forward(p₀),
            other_params,
            opt_bounds;
            noise_scale=s,
        )
        optsol = solve(optprob, optimizer, callback=callback(fout))
    end
end