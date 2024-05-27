using DrWatson
@quickactivate "FenrirForNeuro"
using FenrirForNeuro
using LinearAlgebra, Statistics
using ModelingToolkit
using Optim, OptimizationOptimJL
using DataFrames, CSV

using Random

# set up experiment
# --------------------------------------------
exp_name = splitext(basename(@__FILE__))[1]
out_dir = mkoutdir(exp_name; base_dir="./")
fout = "$out_dir" * "$exp_name.csv"

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

# set loss parameters
other_params =
    (data=data, prob=prob, σ²=σ², proj=proj, order=3, dt=dt_sol, prior=prob_prior)

l(κ²) = loss([θᵢ[2] for θᵢ in θ], (other_params..., logκ²=log.(κ²)))[1]
steps = LinRange(-20, 50, 100)
κ²s = [10.0^i for i in steps]
nll = l.(κ²s)

data = DataFrame([κ²s, nll], [:κ², :nll])

println("min κ²: ", data.κ²[argmin(data.nll)])

CSV.write(fout, data, writeheader=true)
