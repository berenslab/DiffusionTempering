# Hodgkin-Huxley Models
# ----------------------------------------------
# gating variables
αm(V, VT) = -0.32 * (V - VT - 13.0) / (exp(-(V - VT - 13.0) / 4.0) - 1.0)
βm(V, VT) = 0.28 * (V - VT - 40.0) / (exp((V - VT - 40.0) / 5.0) - 1.0)

αn(V, VT) = -0.032 * (V - VT - 15.0) / (exp(-(V - VT - 15.0) / 5.0) - 1.0)
βn(V, VT) = 0.5 * exp(-(V - VT - 10.0) / 40.0)

αh(V, VT) = 0.128 * exp(-(V - VT - 17.0) / 18.0)
βh(V, VT) = 4.0 / (1.0 + exp(-(V - VT - 40.0) / 5.0))

αq(V) = 0.055 * (-27.0 - V) / (exp((-27.0 - V) / 3.8) - 1.0)
βq(V) = 0.94 * exp((-75.0 - V) / 17.0)

αr(V) = 0.000457 * exp((-13.0 - V) / 50.0)
βr(V) = 0.0065 / (exp((-15.0 - V) / 28.0) + 1.0)

τₚ(V, τ_max) = τ_max / (3.3 * exp((V + 35.0) / 20.0) + exp(-(V + 35.0) / 20.0))
τᵤ(V, Vₓ) =
    (30.8 + (211.4 + exp((V + Vₓ + 113.2) / 5.0))) /
    (3.7 * (1.0 + exp((V + Vₓ + 84.0) / 3.2)))

# Would be the solution when dx/dt = 0:
m∞(V, VT) = 1.0 / (1.0 + βm(V, VT) / αm(V, VT))
n∞(V, VT) = 1.0 / (1.0 + βn(V, VT) / αn(V, VT))
h∞(V, VT) = 1.0 / (1.0 + βh(V, VT) / αh(V, VT))
p∞(V) = 1.0 / (1.0 + exp(-(V + 35.0) / 10.0))
q∞(V) = 1.0 / (1.0 + βq(V) / αq(V))
r∞(V) = 1.0 / (1.0 + βr(V) / αr(V))
s∞(V, Vₓ) = 1.0 / (1.0 + exp(-(V + Vₓ + 57.0) / 6.2))
u∞(V, Vₓ) = 1.0 / (1.0 + exp(-(V + Vₓ + 81.0) / 4.0))

function A_cylinder(d=60, L=nothing)
    L = isnothing(L) ? d : L
    d *= 1e-4 # μm -> cm
    L *= 1e-4 # μm -> cm
    r = d / 2
    return 2 * π * r * (r + L)
end

function get_Asoma(τₘ, Rₘ)
    return (
        τₘ * 1e-3 # ms -> s
        / Rₘ # MΩ
    )
end

# parameter bounds
"These ranges closely follow the parameters observed in Pospischil et al. 2008,
which closely follows the set up from Goncalves et al. 2020."
hh_θ_bounds = ComponentVector(
    C=(0.4, 3),
    A=(A_cylinder(20), A_cylinder(80)),
    gNa=(0.5, 80.0),
    ENa=(50, 100),
    gK=(1e-4, 15),
    EK=(-110, -70),
    gleak=(1e-4, 0.6),
    Eleak=(-100, -35),
    VT=(-90, -40),
    gM=(1e-4, 0.6),
    τ_max=(50, 3e3),
    gL=(-1e-4, 0.6),
    ECa=(100, 150),
    gT=(-1e-4, 0.6),
    Vₓ=(0, 4),
    V₀=(-80, -60),
)

# stimulus
# function Iₑ(t)
#     (10 <= t <= 90) * 500e-6 # μA
# end

function Iₑ(t, amp=210)
    if typeof(t) <: Float64
        I(t) = (10 <= t <= 90) * amp # pA
        return I(t) * 1e-6 # μA
    else # Taylor1 or TaylorN
        return 0
    end
end
@register_symbolic Iₑ(t, amp)

function I₀(t)
    return 0
end
@register_symbolic I₀(t)

# Models
"The model is an implementation of Pospischil et al. 2008. The default parameters
are chosen to be similar to the values found in 
http://help.brain-map.org/download/attachments/8323525/BiophysModelPeri.pdf.
A_soma was taken from in vitro recordings from mouse cortex http://celltypes.brain-map.org/
Cell_ID: 509881736
τₘ = 15.3 ms
Rₘ = 184 MΩ.

This procedure closely follows Goncalves et al. 2020."
function FullPospischilHHSystem(;
    A=get_Asoma(15.3, 184),
    C=1,
    gNa=25,
    ENa=53,
    gK=7,
    EK=-107,
    gleak=0.1,
    Eleak=-70,
    VT=-60,
    gM=0.01,
    τ_max=4e3,
    gL=0.01,
    ECa=120,
    gT=0.0,
    Vₓ=2,
    V₀=-70.0,
    Iₑ=Iₑ,
    name=:Pospischil,
)
    kwargs = NamedTuple(Base.@locals)[Base.kwarg_decl.(methods(PospischilHHSystem))[1]]
    vars = [k => v for (k, v) in Base.pairs(kwargs) if k ∉ [:name, :Iₑ]]
    θ = [v for (k, v) in vars if typeof(v) <: Num]

    @variables t
    sts = @variables V(t) = V₀ m(t) = m∞(V₀, VT) h(t) = h∞(V₀, VT) n(t) = n∞(V₀, VT) p(t) =
        p∞(V₀) q(t) = q∞(V₀) r(t) = r∞(V₀) u(t) = u∞(V₀, Vₓ)
    D = Differential(t)

    fm = D(m) ~ (αm(V, VT) * (1 - m) - βm(V, VT) * m)
    fh = D(h) ~ (αh(V, VT) * (1 - h) - βh(V, VT) * h)
    fn = D(n) ~ (αn(V, VT) * (1 - n) - βn(V, VT) * n)
    fp = D(p) ~ (p∞(V) - p) / τₚ(V, τ_max)
    fq = D(q) ~ (αq(V) * (1 - q) - βq(V) * q)
    fr = D(r) ~ (αr(V) * (1 - r) - βr(V) * r)
    fu = D(u) ~ (u∞(V, Vₓ) - u) / τᵤ(V, Vₓ)

    INa = gNa * m^3 * h * (ENa - V)
    IK = gK * n^4 * (EK - V)
    Ileak = gleak * (Eleak - V)
    IM = gM * p * (EK - V)
    IL = gL * q^2 * r * (ECa - V)
    IT = gT * s∞(V, Vₓ)^2 * u * (ECa - V)
    Iin = Iₑ(t)
    fV = D(V) ~ (INa + IK + Ileak + IM + IL + IT + Iin / A) / C

    eqs = [fV, fm, fh, fn, fp, fq, fr, fu]

    return ODESystem(eqs, t, sts, θ; name=name)
end

"The model is a reduced implementation of Pospischil et al. 2008. The default parameters
are chosen to be similar to the values found in 
http://help.brain-map.org/download/attachments/8323525/BiophysModelPeri.pdf.
A_soma was taken from in vitro recordings from mouse cortex http://celltypes.brain-map.org/
Cell_ID: 509881736
τₘ = 15.3 ms
Rₘ = 184 MΩ.

This procedure closely follows Goncalves et al. 2020."
function ReducedPospischilHHSystem(;
    A=get_Asoma(15.3, 184),
    C=1,
    gNa=25,
    ENa=53,
    gK=7,
    EK=-107,
    gleak=0.1,
    Eleak=-70,
    VT=-60,
    gM=0.01,
    τ_max=4e3,
    gL=0.01,
    ECa=120,
    V₀=-70.0,
    Iₑ=Iₑ,
    name=:Pospischil,
)
    kwargs =
        NamedTuple(Base.@locals)[Base.kwarg_decl.(methods(ReducedPospischilHHSystem))[1]]
    vars = [k => v for (k, v) in Base.pairs(kwargs) if k ∉ [:name, :Iₑ]]
    θ = [v for (k, v) in vars if typeof(v) <: Num]

    @variables t
    sts = @variables V(t) = V₀ m(t) = m∞(V₀, VT) h(t) = h∞(V₀, VT) n(t) = n∞(V₀, VT) p(t) =
        p∞(V₀) q(t) = q∞(V₀) r(t) = r∞(V₀)
    D = Differential(t)

    fm = D(m) ~ (αm(V, VT) * (1 - m) - βm(V, VT) * m)
    fh = D(h) ~ (αh(V, VT) * (1 - h) - βh(V, VT) * h)
    fn = D(n) ~ (αn(V, VT) * (1 - n) - βn(V, VT) * n)
    fp = D(p) ~ (p∞(V) - p) / τₚ(V, τ_max)
    fq = D(q) ~ (αq(V) * (1 - q) - βq(V) * q)
    fr = D(r) ~ (αr(V) * (1 - r) - βr(V) * r)

    INa = gNa * m^3 * h * (ENa - V)
    IK = gK * n^4 * (EK - V)
    Ileak = gleak * (Eleak - V)
    IM = gM * p * (EK - V)
    IL = gL * q^2 * r * (ECa - V)
    Iin = Iₑ(t)
    fV = D(V) ~ (INa + IK + Ileak + IM + IL + Iin / A) / C

    eqs = [fV, fm, fh, fn, fp, fq, fr]

    return ODESystem(eqs, t, sts, θ; name=name)
end

"The model is a stripped down implementation of Pospischil et al. 2008. The 
default parameters are chosen to be similar to the values found in 
http://help.brain-map.org/download/attachments/8323525/BiophysModelPeri.pdf.
A_soma was taken from in vitro recordings from mouse cortex http://celltypes.brain-map.org/
Cell_ID: 509881736
τₘ = 15.3 ms
Rₘ = 184 MΩ.

This procedure closely follows Goncalves et al. 2020."
function SimpleHHSystem(;
    C=1,
    A=get_Asoma(15.3, 184),
    gNa=25,
    ENa=53,
    gK=7,
    EK=-107,
    gleak=0.1,
    Eleak=-70,
    VT=-60,
    V₀=-70.0,
    Iₑ=Iₑ,
    name=:HHsimple,
)
    kwargs = NamedTuple(Base.@locals)[Base.kwarg_decl.(methods(SimpleHHSystem))[1]]
    vars = [k => v for (k, v) in Base.pairs(kwargs) if k ∉ [:name, :Iₑ]]
    θ = [v for (k, v) in vars if typeof(v) <: Num]

    @variables t
    sts = @variables V(t) = V₀ m(t) = m∞(V₀, VT) h(t) = h∞(V₀, VT) n(t) = n∞(V₀, VT)
    D = Differential(t)

    fm = D(m) ~ (αm(V, VT) * (1 - m) - βm(V, VT) * m)
    fh = D(h) ~ (αh(V, VT) * (1 - h) - βh(V, VT) * h)
    fn = D(n) ~ (αn(V, VT) * (1 - n) - βn(V, VT) * n)

    INa = gNa * m^3 * h * (ENa - V)
    IK = gK * n^4 * (EK - V)
    Ileak = gleak * (Eleak - V)
    Iin = Iₑ(t)
    fV = D(V) ~ (INa + IK + Ileak + Iin / A) / C

    eqs = [fV, fm, fh, fn]

    return ODESystem(eqs, t, sts, θ; name=name)
end

function get_SinglecompartmentHH(
    θ,
    tspan=(0.0, 100.0);
    constants=Dict(),
    Sys=SimpleHHSystem,
    name=:HHsinglecomp,
)
    params = [Symbol(k) => k for (k, v) in θ]
    sys = Sys(; params..., constants..., name=name)
    sys = structural_simplify(sys)

    prior_bounds = hh_θ_bounds[Symbol.(parameters(sys))]
    prior = Product([Uniform(lb, ub) for (lb, ub) in prior_bounds])

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(sys, [], tspan, θ, jac=true)
    return prior, prob
end

function get_cable_matrix(gs)
    N = length(gs) + 1
    G = zeros(N, N)
    G[diagind(G, -1)] = gs
    G[diagind(G, 1)] = gs
    G[diagind(G)] = -([0 gs...] + [gs... 0])
    return G
end

function couple(compartments, G; name)
    dx2sym(eq) = Symbolics.tosymbol(eq.lhs)
    eqs = [equations(c) for c in compartments]
    x = [dx2sym(eq) for eq in eqs for eq in eq]
    at_V = findall(xᵢ -> xᵢ == Symbol("Vˍt(t)"), x)

    @variables t
    @named comps = compose(compartments...)
    eqs = equations(comps)
    Dx = [eq.lhs for eq in eqs]
    fx = [eq.rhs for eq in eqs]
    Vs = states(comps)[at_V]

    fx[at_V] += G * Vs
    eqs = Dx .~ fx
    return ODESystem(eqs, t; name=name)
end

function get_MulticompartmentHH(θ, compartments, G, tspan=(0.0, 100.0))
    @named HHmulticomp = couple(compartments, G)
    HHmulticomp = structural_simplify(HHmulticomp)

    params = vcat(parameters.(compartments)...)
    is_in(a, b) = occursin(string(a), string(b))
    match_θᵢ_bounds(p) = [b for (k, b) in pairs(NamedTuple(hh_θ_bounds)) if is_in(k, p)][1]
    prior_bounds = [match_θᵢ_bounds(p) for p in params]
    prior = Product([Uniform(lb, ub) for (lb, ub) in prior_bounds])

    θ = vcat([θᵢ for θᵢ in θ]...)
    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(HHmulticomp, [], tspan, θ, jac=true)
    return prior, prob
end

# Lotka Volterra
# ----------------------------------------------

lv_θ_bounds = ComponentVector(
    α=(1e-3, 5.0),
    β=(1e-3, 5.0),
    γ=(1e-3, 5.0),
    δ=(1e-3, 5.0),
    x₀=(0.0, 10.0),
    y₀=(0.0, 10.0),
)

function LotkaVolterraSystem(;
    α=1.5,
    β=1.0,
    γ=3.0,
    δ=1.0,
    x₀=1.0,
    y₀=1.0,
    name=:LotkaVolterra,
)
    kwargs = NamedTuple(Base.@locals)[Base.kwarg_decl.(methods(LotkaVolterraSystem))[1]]
    vars = [k => v for (k, v) in Base.pairs(kwargs) if k ∉ [:name]]
    θ = [v for (k, v) in vars if typeof(v) <: Num]

    @variables t
    sts = @variables x(t) = x₀ y(t) = y₀
    D = Differential(t)

    fx = D(x) ~ α * x - β * x * y
    fy = D(y) ~ δ * x * y - γ * y

    eqs = [fx, fy]

    return ODESystem(eqs, t, sts, θ; name=name)
end

function get_LV(θ, tspan=(0.0, 20.0); constants=Dict())
    params = [Symbol(k) => k for (k, v) in θ]
    @named LV = LotkaVolterraSystem(; params..., constants...)
    LV = structural_simplify(LV)

    prior_bounds = lv_θ_bounds[Symbol.(parameters(LV))]
    prior = Product([Uniform(lb, ub) for (lb, ub) in prior_bounds])

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(LV, [], tspan, θ, jac=true)
    return prior, prob
end

# Pendulum
# ----------------------------------------------

pendulum_θ_bounds = ComponentVector(l=(0.1, 10.0), φ₀=(0.0, 2 * π), g=(1e-3, 20.0))

function PendulumSystem(; l=1, φ₀=π / 4, g=9.81, name=:Pendulum)
    kwargs = NamedTuple(Base.@locals)[Base.kwarg_decl.(methods(PendulumSystem))[1]]
    vars = [k => v for (k, v) in Base.pairs(kwargs) if k ∉ [:name]]
    θ = [v for (k, v) in vars if typeof(v) <: Num]

    @variables t
    sts = @variables φₓ(t) = φ₀ φᵥ(t) = 0
    D = Differential(t)

    fφₓ = D(φₓ) ~ φᵥ
    fφᵥ = D(φᵥ) ~ -g / l * sin(φₓ)

    return ODESystem([fφᵥ, fφₓ], t, sts, θ; name=name)
end

function get_Pendulum(θ, tspan=(0.0, 10.0); constants=Dict())
    params = [Symbol(k) => k for (k, v) in θ]
    @named pendulum = PendulumSystem(; params..., constants...)
    pendulum = structural_simplify(pendulum)

    prior_bounds = pendulum_θ_bounds[Symbol.(parameters(pendulum))]
    prior = Product([Uniform(lb, ub) for (lb, ub) in prior_bounds])

    prob = ODEProblem{true,SciMLBase.FullSpecialize()}(pendulum, [], tspan, θ, jac=true)
    return prior, prob
end

# function get_ode_prob(System, name, θ, θ_bounds, tspan=(0.0, 10.0))
#     sys = System(; [Symbol(k) => k for (k, v) in θ]..., name=name)
#     sys = structural_simplify(sys)

#     prior_bounds = θ_bounds[Symbol.(parameters(sys))]
#     prior = Product([Uniform(lb, ub) for (lb, ub) in prior_bounds])

#     prob = ODEProblem{true,SciMLBase.FullSpecialize()}(sys, [], tspan, θ, jac=true)
#     return prior, prob
# end