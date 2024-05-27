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
@parameters gK
θ = [gK => 7]
prob_prior, prob = get_SinglecompartmentHH(θ)

# set up inference
# --------------------------------------------
σ² = 1e-1
dt_data = 1e-2
dt_sol = 1e-2
proj = [1 0 0 0]
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

ps₀ = get_prior_lattice(100, prior, LinRange)
from, to = length(ARGS) == 2 ? parse.(Int64, ARGS) : (1, size(ps₀, 1))

fout = "$out_dir" * "$exp_name-$from-$to.csv"
println("currently writing to ", fout)
for (idx, p₀) in zip(from:to, eachrow(ps₀[from:to, :]))
    println("$idx/$(to-from+1)")
    l(κ²) = loss(p₀, (other_params..., logκ²=log.(κ²)))[1]

    row = []
    push!(row, p₀...)
    for κ² in tempering_schedule
        try
            push!(row, l(κ²))
        catch e
            push!(row, NaN)
        end
    end
    vec2csv(fout, row)
end