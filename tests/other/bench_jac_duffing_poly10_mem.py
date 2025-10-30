# bench_jac_duffing_poly10_mem.py
# Benchmark jax.jacfwd vs jax.jacrev on a degree-10 polynomial residual
# and profile host memory (RSS) and Python allocations (tracemalloc).
# Author: you + ChatGPT (MIT)

import os
import time
import argparse

import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

import jax
import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


# ---------------------------------
# Quadrature and Legendre utilities
# ---------------------------------

def gauss_legendre(n):
    tau, w = np.polynomial.legendre.leggauss(n)
    return jnp.array(tau, dtype=jnp.float64), jnp.array(w, dtype=jnp.float64)

def precompute_legendre_all(tau, m):
    """
    Compute Legendre P_n, P'_n, P''_n for n=0..m-1 at all tau.
    Uses identities valid on (-1,1), which is fine for Gauss nodes.
    """
    Q = tau.shape[0]
    # P_0 = 1, P_1 = τ
    P = [jnp.ones(Q, dtype=jnp.float64)]
    if m >= 2:
        P.append(tau.astype(jnp.float64))
    for n in range(2, m):
        # n Pn = (2n-1)τ P_{n-1} - (n-1) P_{n-2}
        Pn = ((2*n-1) * tau * P[n-1] - (n-1) * P[n-2]) / n
        P.append(Pn)
    P = jnp.stack(P, axis=1)  # (Q, m)

    denom = (1.0 - tau**2)    # safe since Gauss nodes exclude ±1

    # First derivatives:
    # (1-τ^2) P'_n = n (P_{n-1} - τ P_n)
    P1_list = [jnp.zeros(Q, dtype=jnp.float64)]
    for n in range(1, m):
        P1n = n * (P[:, n-1] - tau * P[:, n]) / denom
        P1_list.append(P1n)
    P1 = jnp.stack(P1_list, axis=1)  # (Q, m)

    # Second derivatives from differentiating the identity:
    # P''_n = [ n P'_{n-1} - n P_n + (2 - n) τ P'_n ] / (1 - τ^2)
    P2_list = [jnp.zeros(Q, dtype=jnp.float64)]
    for n in range(1, m):
        P2n = (n * P1[:, n-1] - n * P[:, n] + (2 - n) * tau * P1[:, n]) / denom
        P2_list.append(P2n)
    P2 = jnp.stack(P2_list, axis=1)  # (Q, m)

    return P, P1, P2


# ---------------------------------
# Interval precompute (single-interval)
# ---------------------------------

def build_interval(m=10, Q=32, T=2.0*np.pi):
    """
    Precompute basis and quadrature for one interval of length T.
    Free modes: ℓ = m (degree 10 default).
    """
    tau, w = gauss_legendre(Q)
    P, P1, P2 = precompute_legendre_all(tau, m)

    # Shape functions that zero value & slope at τ=-1:
    # ψ_k(τ) = (τ+1)^2 P_k(τ)
    S  = (tau + 1.0) ** 2
    S1 = 2.0 * (tau + 1.0)
    S2 = jnp.full_like(tau, 2.0)

    h = jnp.array(T, dtype=jnp.float64)

    psi      = S[:, None] * P
    dpsi_dt  = (2.0/h) * (S1[:, None] * P + S[:, None] * P1)
    d2psi_dt = (2.0/h)**2 * (S2[:, None] * P + 2.0*S1[:, None]*P1 + S[:, None]*P2)

    sqrt_w = jnp.sqrt(w * h / 2.0)

    return {
        "m": m, "Q": Q, "T": jnp.array(T, dtype=jnp.float64), "h": h,
        "tau": tau, "w": w,
        "P": P, "P1": P1, "P2": P2,
        "psi": psi, "dpsi_dt": dpsi_dt, "d2psi_dt": d2psi_dt,
        "sqrt_w": sqrt_w
    }


# ---------------------------------
# Duffing-like residual vector and energy
# ---------------------------------

def make_residual_functions(pre, phys, x_left=0.1, v_left=0.0, phi=0.0):
    """
    Return JITted functions:
      rw(coeffs): weighted residual vector (Q,)
      E(coeffs):  scalar energy = ||rw||^2
    """
    tau = pre["tau"]; h = pre["h"]
    t_q = (tau + 1.0) * h / 2.0  # left time=0 for this single interval
    psi, dpsi_dt, d2psi_dt = pre["psi"], pre["dpsi_dt"], pre["d2psi_dt"]
    sqrt_w = pre["sqrt_w"]

    c = jnp.array(phys["c"], dtype=jnp.float64)
    k = jnp.array(phys["k"], dtype=jnp.float64)
    gamma = jnp.array(phys["gamma"], dtype=jnp.float64)
    f = jnp.array(phys["f"], dtype=jnp.float64)
    omega = jnp.array(phys["omega"], dtype=jnp.float64)
    phi = jnp.array(phi, dtype=jnp.float64)

    # base part enforces x(0)=x_left, ẋ(0)=v_left
    base_x = jnp.array(x_left, dtype=jnp.float64) + jnp.array(v_left, dtype=jnp.float64) * (h / 2.0) * (tau + 1.0)
    base_v = jnp.full_like(tau, v_left, dtype=jnp.float64)
    base_a = jnp.zeros_like(tau, dtype=jnp.float64)

    @jit
    def residual_vector(coeffs):
        X = base_x + psi @ coeffs
        V = base_v + dpsi_dt @ coeffs
        A = base_a + d2psi_dt @ coeffs
        r = A + c * V + k * X + gamma * (X**3) - f * jnp.cos(omega * t_q + phi)
        return sqrt_w * r

    @jit
    def energy(coeffs):
        rw = residual_vector(coeffs)
        return jnp.dot(rw, rw)

    return residual_vector, energy


# ---------------------------------
# Memory helpers
# ---------------------------------

def _arr_nbytes(a) -> int:
    return int(a.size * a.dtype.itemsize)

def bytes_precompute(pre) -> int:
    # Estimate total bytes of stored arrays in 'pre'
    keys = ["tau","w","P","P1","P2","psi","dpsi_dt","d2psi_dt","sqrt_w","h","T"]
    total = 0
    for k in keys:
        a = pre[k]
        total += _arr_nbytes(a)
    return total

def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    kb = n / 1024.0
    if kb < 1024:
        return f"{kb:.2f} KiB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MiB"
    gb = mb / 1024.0
    return f"{gb:.2f} GiB"


# ---------------------------------
# Benchmark + memory profile
# ---------------------------------

def bench_time(fn, arg, repeat=50, label=""):
    # First call includes compilation; steady-state averages exclude compile time.
    t0 = time.perf_counter()
    y = fn(arg)
    jax.block_until_ready(y)
    t1 = time.perf_counter()
    first = t1 - t0

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

def bench_memory(fn, arg, repeat=50, label=""):
    """
    Profile host memory:
      - Peak Python allocations via tracemalloc (approx; excludes device memory)
      - Peak RSS (requires psutil; excludes GPU device memory)
    """
    import tracemalloc
    if not _HAS_PSUTIL:
        print(f"{label:30s}  (psutil not installed; RSS unavailable)")
    else:
        process = psutil.Process(os.getpid())

    # Warm compile
    y = fn(arg)
    jax.block_until_ready(y)

    # Measure over 'repeat' calls
    tracemalloc.start()
    peak_rss = process.memory_info().rss if _HAS_PSUTIL else 0
    out = None
    for _ in range(repeat):
        out = fn(arg)
        jax.block_until_ready(out)
        if _HAS_PSUTIL:
            rss = process.memory_info().rss
            if rss > peak_rss:
                peak_rss = rss
    current, peak_tracemalloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if _HAS_PSUTIL:
        print(f"{label:30s}  peak_tracemalloc={fmt_bytes(int(peak_tracemalloc))}   peak_RSS={fmt_bytes(int(peak_rss))}")
    else:
        print(f"{label:30s}  peak_tracemalloc={fmt_bytes(int(peak_tracemalloc))}")

    return int(peak_tracemalloc), int(peak_rss) if _HAS_PSUTIL else 0


def main():
    parser = argparse.ArgumentParser(description="Benchmark jax.jacfwd vs jax.jacrev (ℓ=10) with memory profiling.")
    parser.add_argument("--m", type=int, default=10, help="Number of free modes ℓ (default 10)")
    parser.add_argument("--Q", type=int, default=32, help="Gauss-Legendre nodes per interval (default 32)")
    parser.add_argument("--repeat", type=int, default=50, help="Repetitions for steady-state timing/memory sampling")
    args = parser.parse_args()

    m = args.m
    Q = args.Q
    repeat = args.repeat

    # Device info
    dev = jax.devices()[0]
    print(f"Using JAX platform: {dev.platform}")
    print(f"Device: {dev}\n")

    # Build precompute and residuals
    pre = build_interval(m=m, Q=Q, T=2.0*np.pi)
    pre_bytes = bytes_precompute(pre)
    print(f"Approx. precompute footprint (host arrays): {fmt_bytes(pre_bytes)}")

    phys = dict(c=1.0/20.0, k=1.0, gamma=0.1, f=0.2, omega=1.0)
    rw_fun, E_fun = make_residual_functions(pre, phys, x_left=0.1, v_left=0.0, phi=0.0)

    # Random coefficients
    key = jax.random.PRNGKey(0)
    coeffs = 0.01 * jax.random.normal(key, (m,), dtype=jnp.float64)

    # JIT baseline funcs
    rw_fun = jit(rw_fun)
    E_fun  = jit(E_fun)

    # Derivative functions
    jac_rw_fwd = jit(jacfwd(lambda c: rw_fun(c)))
    jac_rw_rev = jit(jacrev(lambda c: rw_fun(c)))
    grad_E_fwd = jit(jacfwd(lambda c: E_fun(c)))
    grad_E_rev = jit(jacrev(lambda c: E_fun(c)))

    # Sanity equality
    Jf = jac_rw_fwd(coeffs); jax.block_until_ready(Jf)
    Jr = jac_rw_rev(coeffs); jax.block_until_ready(Jr)
    diffJ = float(jnp.max(jnp.abs(Jf - Jr)))
    print(f"‣ ||J_fwd - J_rev||_max  = {diffJ:.3e}")

    gf = grad_E_fwd(coeffs); jax.block_until_ready(gf)
    gr = grad_E_rev(coeffs); jax.block_until_ready(gr)
    diffg = float(jnp.max(jnp.abs(gf - gr)))
    print(f"‣ ||∇E_fwd - ∇E_rev||_max = {diffg:.3e}\n")

    # Time benchmarks
    print("Timing (first includes compile):")
    bench_time(rw_fun,      coeffs, repeat=2*repeat, label="rw(coeffs) [Q,]")
    bench_time(E_fun,       coeffs, repeat=4*repeat, label="E(coeffs)   [1]")
    bench_time(jac_rw_fwd,  coeffs, repeat=repeat,   label="jacfwd  rw -> J [Q,ℓ]")
    bench_time(jac_rw_rev,  coeffs, repeat=repeat,   label="jacrev  rw -> J [Q,ℓ]")
    bench_time(grad_E_fwd,  coeffs, repeat=2*repeat, label="jacfwd  E  -> g [ℓ]")
    bench_time(grad_E_rev,  coeffs, repeat=2*repeat, label="jacrev  E  -> g [ℓ]")

    print("\nMemory profiling (host):")
    if not _HAS_PSUTIL:
        print("psutil not installed; install with `pip install psutil` to get RSS peaks.\n")

    # Memory benchmarks
    bench_memory(rw_fun,      coeffs, repeat=4*repeat, label="rw(coeffs) [Q,]")
    bench_memory(E_fun,       coeffs, repeat=4*repeat, label="E(coeffs)   [1]")
    bench_memory(jac_rw_fwd,  coeffs, repeat=repeat,   label="jacfwd  rw -> J [Q,ℓ]")
    bench_memory(jac_rw_rev,  coeffs, repeat=repeat,   label="jacrev  rw -> J [Q,ℓ]")
    bench_memory(grad_E_fwd,  coeffs, repeat=2*repeat, label="jacfwd  E  -> g [ℓ]")
    bench_memory(grad_E_rev,  coeffs, repeat=2*repeat, label="jacrev  E  -> g [ℓ]")

    # Optional: try to export device memory profile if available
    try:
        import jax.profiler as jprof
        out_path = "device_memory_profile.pb"
        jprof.save_device_memory_profile(out_path)
        print(f"\nSaved device memory profile to: {out_path}")
        print("You can load this in TensorBoard's 'Memory' plugin for device-side details.")
    except Exception as e:
        print("\n(Device memory profile not saved — feature unavailable on this platform/JAX build.)")


if __name__ == "__main__":
    main()
