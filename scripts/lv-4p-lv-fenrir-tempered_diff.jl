using DrWatson
@quickactivate "FenrirForNeuro"
using FenrirForNeuro
using LinearAlgebra, Statistics
using ModelingToolkit
using Optim, OptimizationOptimJL
using CSV, DataFrames

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
prior = prob_prior
other_params = (data=data, prob=prob, prior=prior, σ²=σ², proj=proj, order=3, dt=dt_sol)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prior)
opt_bounds = forward.(get_prior_bounds(prior))

# set tempering schedule
κ²0 = 20.0
# τ = 5.0; T(t::Real)::Float64 = 10.0^(κ²0 * exp(-t / τ))
T(t::Real)::Float64 = 10.0^(κ²0 - t)
tempering_schedule = T.(LinRange(0, 20, 21))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=true)

# run optimization loop
# --------------------------------------------
ps₀ = rand(prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    sample_idx = lpad(idx, 3, "0")
    fout = "$out_dir" * "$exp_name-$sample_idx.csv"
    iter = 0
    t₀ = time()

    for (i, κ²ᵢ) in enumerate(tempering_schedule)
        optprob = get_optprob(
            opt_loss,
            bump_if_near_bounds(forward(p₀), opt_bounds),
            (other_params..., logκ²=log.(κ²ᵢ)),
            opt_bounds;
            abstol=(i == length(tempering_schedule) ? 1e-9 : 1e-6),
            reltol=(i == length(tempering_schedule) ? 1e-6 : 1e-3),
        )
        try
            optsol = solve(optprob, optimizer, callback=callback(fout, iter, t₀, log.(κ²ᵢ)))
            p₀ = backward(optsol.u)
            orig = optsol.original
            iter = orig.g_calls + orig.iterations + 1
        catch e
            if isa(e, PosDefException)
                @warn "Caught PosDefException"
                df = CSV.read(fout, DataFrame, header=false)
                p_prev = Array(df[end, 1:end-3])
                vec2csv(fout, [p_prev... κ²ᵢ NaN t_elapsed(t₀)])
                p₀ = p_prev
                iter = length(df[:, 1]) + 1
            else
                rethrow(e)
            end
        end
    end
end