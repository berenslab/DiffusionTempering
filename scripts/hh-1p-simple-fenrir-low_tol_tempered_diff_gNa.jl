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
@parameters gNa
θ = [gNa => 25]
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

# set tempering schedule
κ²0 = 20.0
# τ = 5.0; T(t::Real)::Float64 = 10.0^(κ²0 * exp(-t / τ))
T(t::Real)::Float64 = 10.0^(κ²0 - t)
tempering_schedule = T.(LinRange(0, 20, 21))

# define callback
p_names = Symbol.(parameters(prob.f.sys))
p_true = NamedTuple(zip(p_names, prob.p))
function callback(fout=nothing, j₀=0, t₀=time(), logκ²=log.(1e20), tf=backward)
    j = j₀
    losses = []
    function (θ, l, ts, states)
        j += 1

        p_vec = tf(θ)
        logκ² = logκ²

        κ² = exp.(logκ²)
        p_est = NamedTuple(zip(p_names, p_vec))

        @info "[Iteration $j] Callback \n$p_true\n$p_est" κ² l
        push!(losses, l)
        vec2csv(fout, [p_vec... κ² l t_elapsed(t₀)])

        GC.gc() # fixes memory leak in ForwardDiff

        abs_change = abs.(diff(losses))
        change_is_small = abs_change .< 1e-1
        if j > j₀ + 3 && all(change_is_small[end-2:end]) && κ² > tempering_schedule[end]
            @warn "Early stopping at iteration $j"
            return true
        end

        return false
    end
end

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
            abstol=(i == length(tempering_schedule) ? 1e-9 : 1e-1),
            reltol=(i == length(tempering_schedule) ? 1e-6 : 1e-1),
        )
        try
            optsol = solve(optprob, optimizer, callback=callback(fout, iter, t₀, log.(κ²ᵢ)))
            p₀ = backward(optsol.u)
            orig = optsol.original
            iter = orig.g_calls + orig.iterations - 1 + iter
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