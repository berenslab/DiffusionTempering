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
@parameters α β # γ δ
θ = [α => 1.5, β => 1.0]#, γ => 3.0, δ => 1.0]
prob_prior, prob = get_LV(θ)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [1 0]
data = generate_data(prob, proj, σ², dt_data)

# fenrir loss
loss = get_fenrir_loss()

# set loss parameters
prior = prob_prior
other_params = (data=data, prob=prob, prior=prior, σ²=σ², proj=proj, order=3, dt=dt_sol)

# set tempering schedule
κ²0 = 20.0
# τ = 5.0; T(t::Real)::Float64 = 10.0^(κ²0 * exp(-t / τ))
T(t::Real)::Float64 = 10.0^(κ²0 - t)
tempering_schedule = T.(LinRange(0, 20, 21))

ps₀ = get_prior_lattice(30, prior, LinRange)
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))

loss_fout = "$out_dir" * "$(exp_name)_loss_$from-$to.csv"
grad_fout = "$out_dir" * "$(exp_name)_grad_$from-$to.csv"
l(p₀, κ²) = loss(p₀, (other_params..., logκ²=log.(κ²)))[1]
grad(l, p₀, κ²) = ForwardDiff.gradient(x -> l(x, κ²), p₀)

SAVE_GRAD = true
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    loss_row = [p₀...]
    SAVE_GRAD ? grad_row = [p₀...] : nothing
    println("$idx/$(to-from+1): evaluating $p₀")
    for κ² in tempering_schedule
        try
            push!(loss_row, l(p₀, κ²))
        catch e
            if isa(e, InterruptException)
                break
            end
            push!(loss_row, NaN)
        end

        if SAVE_GRAD
            try
                push!(grad_row, grad(l, p₀, κ²)...)
                GC.gc()
            catch e
                push!(grad_row, [NaN, NaN]...)
                GC.gc()
                if isa(e, InterruptException)
                    break
                end
            end
        end
    end
    vec2csv(loss_fout, loss_row)
    SAVE_GRAD ? vec2csv(grad_fout, grad_row) : nothing
end