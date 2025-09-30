# import jax
# jax.config.update("jax_enable_x64", True)
# grad_tanh = jax.grad(jax.numpy.tanh)
# print(grad_tanh(0.2))

# bench_jac_duffing_poly10.py
# Benchmark jax.jacfwd vs jax.jacrev on a single-interval Duffing residual
# using a polynomial with order 10 (ℓ = 10 free modes).
# Author: you + ChatGPT (MIT)

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu") 

# -------------------------------
# Quadrature and Legendre helpers
# -------------------------------

def gauss_legendre(n):
    tau, w = np.polynomial.legendre.leggauss(n)
    return jnp.array(tau, dtype=jnp.float64), jnp.array(w, dtype=jnp.float64)

def precompute_legendre_all(tau, m):
    """
    Legendre P_n, P'_n, P''_n at all tau for n=0..m-1.
    Gauss nodes never include ±1, so divisions by (1-τ^2) are safe.
    """
    Q = tau.shape[0]
    P = [jnp.ones(Q)]
    if m >= 2:
        P.append(tau)
    for n in range(2, m):
        Pn = ((2*n-1) * tau * P[n-1] - (n-1) * P[n-2]) / n
        P.append(Pn)
    P = jnp.stack(P, axis=1)  # (Q, m)

    # First derivative using stable identity:
    # (1-τ^2) P'_n = n (P_{n-1} - τ P_n)
    denom = (1.0 - tau**2)
    P1_list = [jnp.zeros(Q)]
    for n in range(1, m):
        P1n = n * (P[:, n-1] - tau * P[:, n]) / denom
        P1_list.append(P1n)
    P1 = jnp.stack(P1_list, axis=1)  # (Q, m)

    # Second derivative from differentiating the identity:
    # P''_n = [ n P'_{n-1} - n P_n + (2 - n) τ P'_n ] / (1 - τ^2)
    P2_list = [jnp.zeros(Q)]
    for n in range(1, m):
        P2n = (n * P1[:, n-1] - n * P[:, n] + (2 - n) * tau * P1[:, n]) / denom
        P2_list.append(P2n)
    P2 = jnp.stack(P2_list, axis=1)  # (Q, m)

    return P, P1, P2

# -------------------------------
# One-interval Duffing residual, ℓ=10
# -------------------------------

def build_interval(prm):
    """
    Precompute static data for one interval [0, h] with h=T (single-interval test).
    Residual uses Duffing: ẍ + c ẋ + k x + γ x^3 = f cos(ω t + φ).
    """
    # Unpack
    m = prm["m"]              # ℓ free modes
    Q = prm["Q"]              # quadrature points
    T = prm["T"]
    h = T                     # single interval of length T
    tau, w = gauss_legendre(Q)
    P, P1, P2 = precompute_legendre_all(tau, m)

    # Shape functions that zero value & slope at τ=-1:
    # ψ_k(τ) = (τ+1)^2 P_k(τ)  (k=0..m-1)
    S  = (tau + 1.0) ** 2
    S1 = 2.0 * (tau + 1.0)
    S2 = jnp.full_like(tau, 2.0)

    psi      = S[:, None] * P
    dpsi_dt  = (2.0/h) * (S1[:, None] * P + S[:, None] * P1)
    d2psi_dt = (2.0/h)**2 * (S2[:, None] * P + 2.0*S1[:, None]*P1 + S[:, None]*P2)

    # Weighted residual vector multiplier for ∫ r^2 dt over τ∈[-1,1]
    sqrt_w = jnp.sqrt(w * h / 2.0)

    # Right-end evaluation helpers (for completeness; not used in bench)
    n = jnp.arange(m, dtype=jnp.float64)
    psi_right = 4.0 * jnp.ones((m,))
    dpsi_dt_right = (2.0/h) * (4.0 + 2.0 * n * (n + 1.0))

    pre = dict(
        tau=tau, w=w, h=h, T=T, Q=Q, m=m,
        P=P, P1=P1, P2=P2,
        psi=psi, dpsi_dt=dpsi_dt, d2psi_dt=d2psi_dt,
        sqrt_w=sqrt_w,
        psi_right=psi_right, dpsi_dt_right=dpsi_dt_right
    )
    return pre

def make_residual_functions(pre, phys, x_left=0.1, v_left=0.0, phi=0.0):
    """
    Build:
      rw(coeffs): weighted residual vector r_w (shape Q,)
      E(coeffs):  scalar energy sum(r_w^2)
    """
    tau = pre["tau"]; h = pre["h"]
    t_q = (tau + 1.0) * h / 2.0  # since left time = 0
    psi, dpsi_dt, d2psi_dt = pre["psi"], pre["dpsi_dt"], pre["d2psi_dt"]
    sqrt_w = pre["sqrt_w"]

    c = phys["c"]; k = phys["k"]; gamma = phys["gamma"]; f = phys["f"]; omega = phys["omega"]

    # Base (locked) part ensures x(0)=x_left, ẋ(0)=v_left
    base_x = x_left + v_left * (h / 2.0) * (tau + 1.0)
    base_v = jnp.full_like(tau, v_left)
    base_a = jnp.zeros_like(tau)

    @jit
    def residual_vector(coeffs):
        # X,V,A at quadrature nodes
        X = base_x + psi @ coeffs
        V = base_v + dpsi_dt @ coeffs
        A = base_a + d2psi_dt @ coeffs
        r = A + c * V + k * X + gamma * (X**3) - f * jnp.cos(omega * t_q + phi)
        return sqrt_w * r  # r_w  (Q,)

    @jit
    def energy(coeffs):
        rw = residual_vector(coeffs)
        return jnp.dot(rw, rw)

    return residual_vector, energy

# -------------------------------
# Benchmark helpers
# -------------------------------

def bench(fn, arg, repeat=50, label=""):
    """
    Time a compiled function (assumes fn is already jitted).
    Returns (first_call_s, avg_s_per_iter)
    """
    # First call: includes compilation + execution
    t0 = time.perf_counter()
    y = fn(arg)
    # Ensure completion before timing ends
    jax.block_until_ready(y)
    t1 = time.perf_counter()
    first = t1 - t0

    # Repeated calls: measure steady-state execution
    t0 = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = fn(arg)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    avg = (t1 - t0) / repeat

    if label:
        print(f"{label:30s}  first={first*1e3:8.2f} ms   avg={avg*1e3:8.2f} ms")
    return first, avg

def main():
    # Problem / discretization
    ℓ = 10          # polynomial order (free modes)
    Q = 32          # ≥ 2ℓ+3 is nice; 32 is fine for ℓ=10
    T = 2.0 * jnp.pi  # arbitrary; single-interval benchmark over one "period"

    phys = dict(
        c=1.0/20.0,     # c = ω0/Q_f (example)
        k=1.0**2,       # ω0^2
        gamma=0.1,
        f=0.2,
        omega=1.0,
        T=T,
    )
    prm = dict(m=ℓ, Q=Q, T=T)
    pre = build_interval(prm)

    # Build residual vector r_w(c) and energy E(c)
    rw_fun, E_fun = make_residual_functions(pre, phys, x_left=0.1, v_left=0.0, phi=0.0)

    # Random test coefficients (stable scale)
    key = jax.random.PRNGKey(0)
    coeffs = 0.01 * jax.random.normal(key, (ℓ,), dtype=jnp.float64)

    # JIT the residual and energy (so derivative fns only compile their own work)
    rw_fun = jit(rw_fun)
    E_fun  = jit(E_fun)

    # Derivative functions to compare
    # Jacobian of residual vector wrt coeffs: shape (Q, ℓ)
    jac_rw_fwd = jit(jacfwd(lambda c: rw_fun(c)))
    jac_rw_rev = jit(jacrev(lambda c: rw_fun(c)))

    # Gradient of scalar energy wrt coeffs: shape (ℓ,)
    grad_E_fwd = jit(jacfwd(lambda c: E_fun(c)))
    grad_E_rev = jit(jacrev(lambda c: E_fun(c)))

    print("Compiling & benchmarking with ℓ=10, Q=32 ...\n")

    # Sanity: numerical equality (up to fp error)
    Jf = jac_rw_fwd(coeffs); jax.block_until_ready(Jf)
    Jr = jac_rw_rev(coeffs); jax.block_until_ready(Jr)
    diff_J = float(jnp.max(jnp.abs(Jf - Jr)))
    print(f"‣ ||J_fwd - J_rev||_max  = {diff_J:.3e}  (Jacobian of r_w)")

    gf = grad_E_fwd(coeffs); jax.block_until_ready(gf)
    gr = grad_E_rev(coeffs); jax.block_until_ready(gr)
    diff_g = float(jnp.max(jnp.abs(gf - gr)))
    print(f"‣ ||∇E_fwd - ∇E_rev||_max = {diff_g:.3e}  (Gradient of energy)\n")

    # Bench residual and energy themselves (baselines)
    bench(rw_fun, coeffs, repeat=100, label="rw(coeffs) [Q,]")
    bench(E_fun,  coeffs, repeat=200, label="E(coeffs)   [1]")

    # Bench Jacobians / gradients
    bench(jac_rw_fwd, coeffs, repeat=50,  label="jacfwd  rw -> J [Q,ℓ]")
    bench(jac_rw_rev, coeffs, repeat=50,  label="jacrev  rw -> J [Q,ℓ]")
    bench(grad_E_fwd, coeffs, repeat=100, label="jacfwd  E  -> g [ℓ]")
    bench(grad_E_rev, coeffs, repeat=100, label="jacrev  E  -> g [ℓ]")

    print("\nNotes:")
    print(" - For vector output (Q=32) and input dim ℓ=10, forward-mode often wins for J.")
    print(" - For scalar output (E), reverse-mode and forward-mode should be similar;")
    print("   sometimes jacrev is faster for large ℓ.")

if __name__ == "__main__":
    main()
