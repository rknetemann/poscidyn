#!/usr/bin/env julia
# duffing_frequency_response.jl  (BifurcationKit 0.4‑compatible, final)
#
# Frequency‑response curve of the forced Duffing oscillator using
# **BifurcationKit.jl** without relying on experimental APIs.  To avoid the
# “arity” complaint (vector field must be `F!(du,u,p)`), we:
#   • embed the forcing phase variables (cosφ, sinφ) ⇒ autonomous system
#   • create thin wrappers `duffing_aut!`, `jac_aut!` that drop the `t`
#     argument expected by OrdinaryDiffEq.
#
# Pipeline
#   1. Integrate the extended ODE with OrdinaryDiffEq to land on the limit
#      cycle for an initial drive frequency ω₀.
#   2. Use that trajectory to initialise a `PeriodicOrbitOCollProblem`.
#   3. Run pseudo‑arclength continuation sweeping ω and store |x|max.
#   4. Plot the frequency‑response curve.
#
# ---------------------------------------------------------------------------

using LinearAlgebra, StaticArrays, OrdinaryDiffEq
using BifurcationKit                   # v0.4.x series
const BK = BifurcationKit
using Plots

# ----------------‑ Duffing parameters --------------------------------------
const delta = 0.22         # damping
const alpha = -1.0         # linear stiffness (double‑well if < 0)
const beta  =  1.0         # cubic stiffness
const Famp  =  0.3         # forcing amplitude

ω_min, ω_max = 0.2, 2.0    # continuation sweep range
ω₀           = 0.8          # seed drive frequency
const IDX_W  = 1            # parameter index inside p (must be Int)

# ----------------‑ 4‑D autonomous vector field -----------------------------
# State: u = [x, v, c, s] := [x, ẋ, cosφ, sinφ]

function duffing_ext!(du, u, p, t)
    x, v, c, s = u
    ω = p[IDX_W]
    du[1] = v
    du[2] = -delta*v - alpha*x - beta*x^3 + Famp*c
    du[3] = -ω*s
    du[4] =  ω*c
    return nothing
end

function jac_ext!(J, u, p, t)
    x, v, c, s = u
    ω = p[IDX_W]
    @views begin
        J[1,1] = 0.0;                    J[1,2] = 1.0;   J[1,3] = 0.0;  J[1,4] = 0.0
        J[2,1] = -alpha - 3beta*x^2;     J[2,2] = -delta; J[2,3] = Famp; J[2,4] = 0.0
        J[3,1] = 0.0;                    J[3,2] = 0.0;   J[3,3] = 0.0;  J[3,4] = -ω
        J[4,1] = 0.0;                    J[4,2] = 0.0;   J[4,3] =  ω;   J[4,4] = 0.0
    end
    return nothing
end

# --- Thin 3‑arg wrappers accepted by BifurcationKit (F!(du,u,p)) ------------

duffing_aut!(du, u, p) = duffing_ext!(du, u, p, 0.0)
function jac_aut!(J, u, p)
    jac_ext!(J, u, p, 0.0)
end

# ----------------‑ Stage 1: integrate to get seed orbit --------------------
println("\n>>> Integrating extended Duffing system to build seed orbit …")

u0   = [0.0, 0.0, 1.0, 0.0]              # start at (x,v) = (0,0), φ = 0 ⇒ (c,s) = (1,0)
p0   = [ω₀]
T₀   = 2π / ω₀                           # expected fundamental period
Tsim = 200T₀                             # long run to kill transients

odeprob = ODEProblem(duffing_ext!, u0, (0.0, Tsim), p0)
sol     = solve(odeprob, Vern9(); reltol=1e-10, abstol=1e-12)
useed   = sol.u[end]
println("Seed point: x=$(useed[1]), v=$(useed[2]), c=$(useed[3]), s=$(useed[4])")

# ----------------‑ Stage 2: collocation problem ---------------------------
par_idx = IDX_W
bifprob = BK.BifurcationProblem(duffing_aut!, useed, p0, par_idx;
                                 J! = jac_aut!,   # analytic Jacobian (in‑place)
                                 inplace = true)

po_algo = BK.PeriodicOrbitOCollProblem(40, 4; jacobian = BK.DenseAnalyticalInplace())
println(">>> Generating collocation initial guess …")
probpo, ci_state = BK.generate_ci_problem(po_algo, bifprob, sol, T₀)

# ----------------‑ Stage 3: continuation parameters -----------------------
opts = BK.ContinuationPar(
    ds = 1e-2, dsmin = 1e-3, dsmax = 5e-2,
    p_min = ω_min, p_max = ω_max,
    max_steps = 300,
    newton_options = BK.NewtonPar(tol = 1e-11, max_iterations = 20),
    detect_bifurcation = 0,
)

# recorder: store |x|max for each periodic orbit
recorder = (
    # extract the displacement row (row 3 == x) from the collocation orbit
    record_from_solution = (x, opt; k...) -> begin
        xt = BK.get_periodic_orbit(opt.prob, x, opt.p)  # matrix: state × time
        amp = maximum(abs.(xt[3, :]))                   # |x|_max  ← displacement row
        (; amp = amp, p = opt.p)
    end,
)

# ----------------‑ Stage 4: run continuation ------------------------------
println(">>> Running pseudo‑arclength continuation in ω …")
branch = BK.continuation(probpo, ci_state, BK.PALC(tangent = BK.Bordered()), opts;
                         verbosity = 2, bothside = true, plot=false,
                         linear_algo = BK.COPBLS(), recorder...)
println(">>> Continuation finished – computed $(length(branch)) periodic orbits.")

# ----------------‑ Stage 5: plot frequency‑response -----------------------
ω_vals  = branch.branch.p
amps    = branch.branch.amp

plot(ω_vals, amps;
     xlabel    = "Driving frequency ω",
     ylabel    = "Response amplitude |x|max",
     linewidth = 2, label = nothing,
     title     = "Duffing oscillator – frequency response")

savefig("duffing_frequency_response.png")
println("Figure saved to duffing_frequency_response.png")
