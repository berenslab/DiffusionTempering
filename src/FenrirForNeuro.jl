module FenrirForNeuro

using Reexport

@reexport using ComponentArrays
@reexport using Distributions

using OrdinaryDiffEq
using UnPack
using ProbNumDiffEq
using SciMLBase
using Flux: sigmoid
using Fenrir
using Plots
using LinearAlgebra
using Optimization
using ForwardDiff
using ModelingToolkit
using Symbolics

include("models.jl")
include("utils.jl")
include("plot.jl")
export A_cylinder,
    get_Asoma,
    Iₑ,
    I₀,
    FullPospischilHHSystem,
    ReducedPospischilHHSystem,
    SimpleHHSystem,
    get_LV,
    get_Pendulum,
    PendulumSystem,
    LotkaVolterraSystem,
    get_SinglecompartmentHH,
    get_MulticompartmentHH,
    get_cable_matrix,
    couple,
    vec2csv,
    get_fenrir_loss,
    get_l2_loss,
    standardize_loss,
    get_std_param_tf,
    merge_priors,
    get_prior_bounds,
    generate_data,
    unpack_loss_args,
    centered_logrange,
    centered_linrange,
    bump_if_near_bounds,
    push_cache!,
    get_optprob,
    # plot_optim_hist,
    # plot_hh_solution,
    # plot_hh_solution!,
    # plot_optim_iter,
    # animate_optim,
    # optim_replay,
    get_prior_lattice,
    t_elapsed,
    get_rk4_l2_callback,
    get_fenrir_callback,
    get_rk4_diffevo_callback,
    mkoutdir,
    rel_pRMSE,
    tRMSE,
    simulate
end
