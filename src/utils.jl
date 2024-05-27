normalize(x, b, u) = (x .- b) ./ (u .- b)
unnormalize(x, b, u) = b .+ (u .- b) .* x
project(A, p) = p * A

function project(
    x::Union{
        AbstractVector{Gaussian{Vector{Float64},PSDMatrix{Float64,Matrix{Float64}}}},
        Vector{Vector{Float64}},
    },
    proj=[1 0 0 0],
)
    x = [project(u, proj) for u in x]
end

function uncon2con(x, b, u)
    y = sigmoid(x)
    return unnormalize(y, b, u)
end

function con2uncon(y, b, u)
    x = normalize(y, b, u)
    inv_sig(y) = log(y / (1 - y))
    return inv_sig(x)
end

function get_std_param_tf(
    lowerbounds::Vector{Float64},
    upperbounds::Vector{Float64},
    which=1,
)
    if which == 1
        return x -> normalize.(x, lowerbounds, upperbounds),
        x -> unnormalize.(x, lowerbounds, upperbounds)
    elseif which == 2
        return x -> con2uncon.(x, lowerbounds, upperbounds),
        x -> uncon2con.(x, lowerbounds, upperbounds)
    elseif which == 3
        return x -> 2 * normalize.(x, lowerbounds, upperbounds) - 1,
        x -> unnormalize.(1 / 2 * (x + 1), lowerbounds, upperbounds)
    else
        error("which must be 1, 2 or 3")
    end
end

function get_std_param_tf(prior::ContinuousDistribution, which=1)
    get_std_param_tf(Array(minimum(prior)), Array(maximum(prior)), which)
end

function standardize_loss(
    loss::Function,
    lowerbounds::Vector{Float64},
    upperbounds::Vector{Float64},
    return_maps::Bool=true;
    which=1,
)
    # takes params and difs from 0 to 1 instead of from b to u
    forward, backward = get_std_param_tf(lowerbounds, upperbounds, which)
    std_loss(x, other_params) = loss(backward(x), other_params)
    if return_maps
        return std_loss, forward, backward
    else
        return std_loss
    end
end

function standardize_loss(
    loss::Function,
    prior::ContinuousDistribution,
    return_maps::Bool=true;
    which=1,
)
    # takes params and difs from 0 to 1 instead of from b to u
    standardize_loss(
        loss,
        Array(minimum(prior)),
        Array(maximum(prior)),
        return_maps;
        which=which,
    )
end

function get_prior_bounds(prior::ContinuousDistribution)
    lbounds = minimum(prior)
    ubounds = maximum(prior)
    return lbounds, ubounds
end

function merge_priors(priors)
    priors = [isa(p, Product) ? p.v : [p] for p in priors]
    priors = vcat(priors...)
    merged_prior = Product(priors)
    return merged_prior
end

function vec2csv(fout, vec)
    if !isnothing(fout)
        open(fout, "a") do io
            write(io, join(vec, ", ") * "\n")
        end
    end
end

function simulate(prob, dt)
    tₛ, tₑ = prob.tspan
    sol = solve(
        prob,
        RadauIIA5(),
        abstol=1e-14,
        reltol=1e-14,
        dense=false,
        saveat=tₛ:dt:tₑ,
        dt=dt,
    )
    return sol.t, ProbNumDiffEq.stack(sol.u)
end

function generate_data(prob, proj, σ², dt)
    tₛ, tₑ = prob.tspan
    sol = solve(
        prob,
        RadauIIA5(),
        abstol=1e-14,
        reltol=1e-14,
        dense=false,
        saveat=tₛ:dt:tₑ,
        dt=dt,
    )
    t_obs = sol.t[1:end]
    u_obs = [proj * (u + sqrt.(σ²) .* randn(size(u))) for u in sol.u]
    data = (t=t_obs, u=u_obs)
    return data
end

function get_fenrir_loss()
    function nll(θ, other_params)
        @unpack data, prob, proj, σ², order, dt = other_params
        prior = :prior in keys(other_params) ? other_params[:prior] : nothing

        if :logκ² in keys(other_params)
            @unpack logκ² = other_params
            θₚ = θ
        elseif :θₚ in keys(other_params)
            @unpack θₚ = other_params
            logκ² = θ[end]
            prob = remake(prob, u0=eltype(logκ²).(prob.u0))
        else
            θₚ, logκ² = θ[1:end-1], θ[end]
        end
        κ² = exp.(logκ²)

        prob = remake(prob, p=θₚ)

        if :solver in keys(other_params)
            @unpack solver = other_params
            diffmodel =
                κ² isa Number ? FixedDiffusion(κ², false) : FixedMVDiffusion(κ², false)
            alg = solver(order=order, diffusionmodel=diffmodel, smooth=true)
            nll, t_est, u_est = fenrir_nll(prob, data, alg, σ²; proj=proj, dt=dt)
        else
            nll, t_est, u_est =
                fenrir_nll(prob, data, σ², κ²; proj=proj, dt=dt, order=order)
        end
        if !isnothing(prior)
            nll -= logpdf(prior, θ)
        end
        return nll, t_est, u_est
    end
    return nll
end

function get_l2_loss()
    function loss(θ, other_params)
        @unpack prob, data, proj = other_params
        t_data, u_data = data
        dt = t_data[2] - t_data[1]

        sol = solve(remake(prob, p=θ), RadauIIA5(), saveat=t_data, dt=dt)

        if sol.retcode != :Success
            return 9_999_999, nothing, nothing
        end

        u_est = ProbNumDiffEq.stack(sol.u) * proj'
        l = sum(abs2, ProbNumDiffEq.stack(u_data) - u_est) / length(sol)

        return l, sol.t, sol.u
    end
end

function centered_logrange(start, mid, stop, num_points)
    first_half = Int(floor(num_points / 2)) + (num_points % 2)
    second_half = Int(floor(num_points / 2))
    geomspace(s, e, n) = 10 .^ (range(log10(s), log10(e), n))
    logrange = n -> ((geomspace(1, 10, n) .- 1) ./ 9)
    shifted_logrange = (s, e, n) -> (e.-(e.-s).*logrange(n))[2:end]
    return vcat(
        reverse(shifted_logrange(start, mid, first_half)),
        [mid],
        shifted_logrange(stop, mid, second_half + 1),
    )
end

function centered_linrange(start, mid, stop, num_points)
    first_half = Int(floor(num_points / 2)) + (num_points % 2)
    second_half = Int(floor(num_points / 2))
    return vcat(
        LinRange(start, mid, first_half),
        LinRange(mid, stop, second_half + 1)[2:end],
    )
end

function push_cache!(cache, p)
    return [cache; reshape(p, 1, length(p))]
end

function get_optprob(loss, p₀, other_params, bounds=nothing; abstol=1e-9, reltol=1e-6)
    f = OptimizationFunction(loss, Optimization.AutoForwardDiff())
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

function get_prior_lattice(
    npoints::Tuple,
    prior,
    range_iter=LinRange,
    padding=nothing;
    flatten=true,
)
    bounds = get_prior_bounds(prior)
    padding = isnothing(padding) ? zeros(length(bounds)) : padding
    bounds = bounds[1] .+ padding[1], bounds[2] .- padding[2]

    if !applicable(length, range_iter)
        range_iter = collect((range_iter for _ in bounds[1]))
    end

    points = [iter(lb, ub, n) for (lb, ub, iter, n) in zip(bounds..., range_iter, npoints)]

    points = collect(Iterators.product(points...))
    if flatten
        points = ProbNumDiffEq.stack([[x...] for x in points])
    end
    return points
end

function get_prior_lattice(
    npoints::Int64,
    prior,
    range_iter=LinRange,
    padding=nothing;
    flatten=true,
)
    return get_prior_lattice(
        (npoints, npoints),
        prior,
        range_iter,
        padding;
        flatten=flatten,
    )
end

function t_elapsed(t₀)
    return time() - t₀
end

function get_fenrir_callback(prob, tf=x -> x; logκ²_as_kwarg=true)
    p_names = Symbol.(parameters(prob.f.sys))
    p_true = NamedTuple(zip(p_names, prob.p))
    function callback(fout=nothing, j₀=0, t₀=time(), logκ²=log.(1e20))
        j = j₀
        function (θ, l, ts, states)
            j += 1

            if logκ²_as_kwarg
                p_vec = tf(θ)
                logκ² = logκ²
            else
                p_vec, logκ² = tf(θ) |> x -> (x, pop!(x))
            end

            κ² = exp.(logκ²)
            p_est = NamedTuple(zip(p_names, p_vec))

            @info "[Iteration $j] Callback \n$p_true\n$p_est" κ² l
            vec2csv(fout, [p_vec... κ² l t_elapsed(t₀)])

            GC.gc() # fixes memory leak in ForwardDiff
            return false
        end
    end
    return callback
end

function get_rk4_l2_callback(prob, tf=x -> x)
    p_names = Symbol.(parameters(prob.f.sys))
    p_true = NamedTuple(zip(p_names, prob.p))
    function callback(fout=nothing, j₀=0, t₀=time())
        j = j₀
        function (θ, l, ts, states)
            j += 1

            p_vec = tf(θ)
            p_est = NamedTuple(zip(p_names, p_vec))

            @info "[Iteration $j] Callback \n$p_true\n$p_est" l
            vec2csv(fout, [p_vec... l t_elapsed(t₀)])
            return false
        end
    end
    return callback
end

function get_rk4_diffevo_callback(prob, tf=x -> x)
    p_names = Symbol.(parameters(prob.f.sys))
    p_true = NamedTuple(zip(p_names, prob.p))
    function callback(fout=nothing, j₀=0, t₀=time())
        j = j₀
        function (θ, l)
            j += 1
            θ = tf.(θ)
            μ_p = mean(θ)
            σ_p = std(θ)
            num_evals = length(θ) * j
            μ_est = NamedTuple(zip(p_names, μ_p))
            σ_est = NamedTuple(zip(p_names, σ_p))
            @info "[Iteration $j #Evals $num_evals] Callback" p_true μ_est σ_est l

            zipped_est = vcat([[μ, σ] for (μ, σ) in zip(μ_est, σ_est)]...)
            vec2csv(fout, [zipped_est... l num_evals t_elapsed(t₀)])

            return false
        end
    end
    return callback
end

function mkoutdir(exp_name; base_dir="./")
    subdirs = split(exp_name, "-")
    out_dir = joinpath(base_dir, subdirs...) * "/"
    mkpath(out_dir)
    return out_dir
end

function bump_if_near_bounds(x, bounds, abstol=1e-4)
    below_tol = Array(abs.(hcat(bounds...) .- x) .< abstol)
    dx = ones(2, length(x)) .* [1, -1] * abstol
    x = diag(below_tol * dx) + x
end

function tRMSE(y, y_hat)
    MSE = sum(abs2, y - y_hat) / size(y, 1)
    RMSE = sqrt(MSE)
    return RMSE
end

function tRMSE(θ, θ_hat, other_params)
    @unpack prob, data, proj, dt = other_params
    t_obs, u_obs = simulate(remake(prob, p=θ), dt)
    t_est, u_est = simulate(remake(prob, p=θ_hat), dt)

    return tRMSE(u_obs * proj', u_est * proj')
end

function rel_pRMSE(θ_hat, θ)
    θ_hat = ndims(θ_hat) == 1 ? reshape(θ_hat, length(θ_hat), 1) : θ_hat
    MSE = sum(abs2, (θ_hat .- θ) ./ θ) / size(θ, 1)
    RMSE = sqrt.(MSE)
    return RMSE
end

function num_correct_params(θ, θ_hat, thresh=1e-1)
    is_correct = abs.((θ_hat .- θ) ./ θ) < thresh
    return sum(is_correct, dims=2)
end
