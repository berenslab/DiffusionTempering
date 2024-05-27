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
prior = prob_prior
other_params = (data=data, prob=prob, prior=prior, σ²=σ², proj=proj, order=3, dt=dt_sol)

# normalize and bound loss inputs
opt_loss, forward, backward = standardize_loss(loss, prior)
opt_bounds = forward.(get_prior_bounds(prior))

# define callback
callback = get_fenrir_callback(prob, backward; logκ²_as_kwarg=true)

# set tempering schedule
κ²0 = 20.0
T10(t::Real)::Float64 = 10.0^(κ²0 - t)
# Texp(t::Real, τ::Real=5.0)::Float64 = 10.0^(κ²0 * exp(-t / τ))
tempering_schedules = [T10.(LinRange(0, 20, n_steps)) for n_steps in 5:21]

ps₀ = rand(prior, 100)'
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))
for (iₛ, schedule) in enumerate(tempering_schedules)
    # run optimization loop
    # --------------------------------------------
    for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
        sample_idx = lpad(idx, 3, "0")
        fout = "$out_dir" * "$exp_name-$iₛ-$sample_idx.csv"
        iter = 0
        t₀ = time()

        for (s, Tₛ) in enumerate(schedule)
            optprob = get_optprob(
                opt_loss,
                bump_if_near_bounds(forward(p₀), opt_bounds),
                (other_params..., logκ²=log.(Tₛ)),
                opt_bounds;
                abstol=(s == length(schedule) ? 1e-9 : 1e-6),
                reltol=(s == length(schedule) ? 1e-6 : 1e-3),
            )
            try
                optsol =
                    solve(optprob, optimizer, callback=callback(fout, iter, t₀, log.(Tₛ)))
                p₀ = backward(optsol.u)
                orig = optsol.original
                iter = orig.g_calls + orig.iterations + 1
            catch e
                if isa(e, PosDefException)
                    @warn "Caught PosDefException"
                    df = CSV.read(fout, DataFrame, header=false)
                    p_prev = Array(df[end, 1:end-3])
                    vec2csv(fout, [p_prev... Tₛ NaN t_elapsed(t₀)])
                    p₀ = p_prev
                    iter = length(df[:, 1]) + 1
                else
                    rethrow(e)
                end
            end
        end
    end
end