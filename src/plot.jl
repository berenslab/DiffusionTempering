function plot_optim_hist(params, l, param_prior; true_params=[20, 15])
    get_idxs_pairs(M) = Iterators.product(1:length(M[1, :]), 1:length(M[1, :]))
    # lower_trig(M) = M[tril!(trues(size(M)), -1)]
    get_first_row(M) = M[1, 2:end]
    lbounds = minimum(param_prior)
    ubounds = maximum(param_prior)

    # compute pairwise permutations of parameters
    param_pairs = [i for i in get_idxs_pairs(params)]
    param_pairs = get_first_row(param_pairs)
    p_labels = keys(lbounds)

    ps = []
    for (i, j) in param_pairs
        plt = plot(params[:, i], params[:, j], label="approx params", marker=:circle)
        scatter!([true_params[i]], [true_params[j]], label="true params", markersize=[3])
        plot!(xlims=(lbounds[i], ubounds[i]), ylims=(lbounds[j], ubounds[j]))
        xlabel!(string(p_labels[i]))
        ylabel!(string(p_labels[j]))
        push!(ps, plt)
    end

    p2 = plot(l, label="")
    xlabel!("iteration")
    ylabel!("nll")
    push!(ps, p2)

    return ps
end

# function plot_optim_conv(params, l, param_prior; true_params=[20, 15])
#     lbounds = minimum(param_prior)
#     ubounds = maximum(param_prior)

#     p_labels = keys(lbounds)

#     ps = []
#     hline([0], label="true params", color=:black)
#     normalized_params = (params - true_params) / (lbounds - ubounds)
#     plot(normalized_params, label=p_labels)

#     xlabel!(string(p_labels[i]))
#     ylabel!(string(p_labels[j]))
#     push!(ps, plt)

#     p2 = plot(l, label="")
#     xlabel!("iteration")
#     ylabel!("nll")
#     push!(ps, p2)

#     return ps
# end

function plot_optim_hist(params, κ², l, param_prior; true_params=[20, 15])
    ps = plot_optim_hist(params, l, param_prior, true_params=true_params)
    p3 = plot(log10.(κ²), label="")
    xlabel!("iteration")
    ylabel!("log_10(κ²)")
    push!(ps, p3)
end

function plot_hh_solution!(t, x; kwargs...)
    plt = plot!(t, x, marker=:o, fillalpha=0.5, grid=false, markersize=1; kwargs...)
    plot!(ylims=(-110, 80))
    xlabel!("t (ms)")
    ylabel!("U (mV)")
    return plt
end

function plot_hh_solution!(t, x::Vector{Vector{Float64}}; kwargs...)
    x = ProbNumDiffEq.stack(x)
    return plot_hh_solution!(t, x; kwargs...)
end

function plot_hh_solution(t, x; kwargs...)
    plot()
    return plot_hh_solution!(t, x; kwargs...)
end

function plot_hh_solution!(
    t,
    x::AbstractVector{Gaussian{Vector{Float64},PSDMatrix{Float64,Matrix{Float64}}}};
    kwargs...,
)
    means = ProbNumDiffEq.stack([x_i.μ for x_i in x])
    stds = ProbNumDiffEq.stack([sqrt.(diag(x_i.Σ)) for x_i in x])
    return plot_hh_solution!(t, means; ribbon=2stds, kwargs...)
end

function plot_optim_iter(t_o, x_o, t_est, x_est, prob, p_est; proj=[1 0 0 0], kwargs...)
    plt = plot_optim_iter(t_o, x_o, t_est, x_est; proj=proj, kwargs...)
    raw_sol = solve(remake(prob, p=p_est), Tsit5(), dense=false)
    x = project(raw_sol.u, proj)
    plot_hh_solution!(raw_sol.t, x; label="Raw PN Solution", kwargs...)

    return plt
end

function plot_optim_iter(t_o, x_o, t_est, x_est; proj=[1 0 0 0], kwargs...)
    plt = scatter(t_o, ProbNumDiffEq.stack(x_o), label="Data")
    plot_hh_solution!(t_est, project(x_est, proj); label="PN Posterior", kwargs...)
    return plt
end

# function get_callback_iter(loss, other_params, t_obs, x_obs, prior)
#     @unpack prob, logκ² = other_params

#     function plot_iter(ps, κ²s, ls)
#         Channel() do channel
#             for (j, (p, κ², l)) in enumerate(zip(ps, κ²s, ls))
#                 l, t_est, x_est = loss(ComponentArray(p=p), (prob=prob, logκ²=log.(κ²)))
#                 put!(
#                     channel,
#                     plot_func(
#                         p,
#                         κ²,
#                         l,
#                         ps[1:j],
#                         κ²s[1:j],
#                         ls[1:j],
#                         t_est,
#                         x_est,
#                         t_obs,
#                         x_obs,
#                         prior,
#                     ),
#                 )
#             end
#         end
#     end
# end

# function animate_optim(callback_replay_iter, fout=nothing, fps=3)
#     # if isnothing(fout)
#     #     fout = "./solve"
#     #     for k in keys(p0)
#     #         fout *= "-" * string(k) * "-" * @sprintf("%.1f", p0[k])
#     #     end
#     #     fout *= ".csv"
#     # end

#     a = Animation()
#     for plt in callback_replay_iter
#         frame(a, plt)
#     end
#     gif(a, fout, fps=fps)
# end

# function optim_replay(xs, κ²s, ls, loss, prob, callback, delay=0.1)
#     if length(fieldnames(typeof(callback))) > 0
#         cb = callback
#     else
#         cb = callback(xs[1], (prob=prob, logκ²=log.(κ²s[1])), true)
#     end
#     Channel() do channel
#         for (j, (x, κ², l)) in enumerate(zip(xs, κ²s, ls))
#             l = loss(x, (prob=prob, logκ²=log.(κ²)))
#             cb(x, l...)
#             sleep(delay)
#             plt = current()
#             put!(channel, plt)
#         end
#     end
# end

# function optim_replay(xs, ls, loss, prob, callback, delay=0.1)
#     if length(fieldnames(typeof(callback))) > 0
#         cb = callback
#     else
#         cb = callback(xs[1], (prob=prob,), true)
#     end
#     Channel() do channel
#         for (j, (x, l)) in enumerate(zip(xs, ls))
#             l = loss(x, (prob=prob,))
#             cb(x, l...)
#             sleep(delay)
#             plt = current()
#             put!(channel, plt)
#         end
#     end
# end