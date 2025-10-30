# duffing_multistart_jax.py
import math
import itertools
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from functools import partial

# Use 64-bit for accuracy
jax.config.update("jax_enable_x64", True)

# Your JAX BVP solver (the big file you pasted) must be in the same folder.
from jax_solve_bvp import solve_bvp as jax_solve_bvp


# ------------------------------------------------------------
# Duffing setup
# ------------------------------------------------------------
def make_duffing_fun(Q, omega0, gamma, omega, f):
    """RHS fun(t, y) for Duffing with forcing cos(omega t)."""
    def fun(t, y):
        return jnp.array([
            y[1],
            -(omega0 / Q) * y[1] - (omega0 ** 2) * y[0] - gamma * y[0] ** 3 + f * jnp.cos(omega * t)
        ])
    return fun


def periodic_bc(ya, yb):
    """Periodic boundary conditions y(0)=y(T) for [x, v]."""
    return jnp.array([ya[0] - yb[0], ya[1] - yb[1]])


def linear_amplitude(Q, omega0, omega, f):
    """Amplitude of the linear oscillator (used to scale multistart seeds)."""
    den = jnp.sqrt((omega0**2 - omega**2)**2 + (omega * omega0 / Q)**2)
    return f / den


# ------------------------------------------------------------
# Initial guesses (amplitude & phase sweep)
# ------------------------------------------------------------
def build_seed_batch(T, omega, A_lin, m_nodes, num_starts=50,
                     amp_range=(0.2, 3.0), n_phase=10, key=0):
    """
    Build batch of initial guesses y0: shape (B, m_nodes, 2),
    by sweeping amplitude (multiples of A_lin) and phase ∈ [0, 2π).
    """
    B = num_starts
    t = jnp.linspace(0.0, T, m_nodes)

    nP = min(n_phase, B)
    nA = max(1, math.ceil(B / nP))
    amp_scales = jnp.linspace(amp_range[0], amp_range[1], nA)
    phases = jnp.linspace(0.0, 2 * jnp.pi, nP, endpoint=False)

    pairs = list(itertools.product(np.array(amp_scales), np.array(phases)))
    if len(pairs) > B:
        pairs = pairs[:B]

    y0_list = []
    for scale, phi in pairs:
        A0 = float(scale) * float(A_lin)
        x0 = A0 * jnp.cos(omega * t + phi)
        v0 = -A0 * omega * jnp.sin(omega * t + phi)
        y0_list.append(jnp.stack([x0, v0], axis=-1))  # (m_nodes, 2)

    y0_batch = jnp.stack(y0_list, axis=0)                  # (B, m, 2)
    x_batch = jnp.broadcast_to(t, (y0_batch.shape[0], t.shape[0]))  # (B, m)
    return x_batch, y0_batch


# ------------------------------------------------------------
# Vectorized solve with JAX (closure captures fun/bc; no non-array args to jit)
# ------------------------------------------------------------
def make_batch_solver(fun, bc, tol=1e-6, bc_tol=1e-6, max_iterations=10):
    """Return a batched, JIT-compiled solver that vmaps over seeds."""

    def _solve_one(x, y0):
        # fun & bc are captured in the closure, not passed as args
        return jax_solve_bvp(fun, bc, x, y0,
                             tol=tol, bc_tol=bc_tol,
                             max_iterations=max_iterations, jit=True)

    # vmap over (x_batch, y0_batch); jit the whole batch
    batched = jax.jit(jax.vmap(_solve_one, in_axes=(0, 0)))
    return batched


# ------------------------------------------------------------
# Dedup & resample (NumPy for convenience)
# ------------------------------------------------------------
def resample_to_uniform(xs, ys, n_dense=2000):
    """
    xs: (B, m) times, ys: (B, m, 2) states -> common t_dense: (n,), y_dense: (B, n, 2)
    """
    xs_np = np.asarray(xs)
    ys_np = np.asarray(ys)
    B, m, d = ys_np.shape
    assert d == 2

    # Use median period to build a common grid (they should all be ~T)
    T_med = float(np.median(xs_np[:, -1]))
    t_dense = np.linspace(0.0, T_med, n_dense)
    y_dense = np.empty((B, n_dense, d), dtype=np.float64)

    for i in range(B):
        ti = xs_np[i]
        xi = ys_np[i, :, 0]
        vi = ys_np[i, :, 1]
        # ensure strictly increasing
        mask = np.concatenate(([True], np.diff(ti) > 0))
        ti, xi, vi = ti[mask], xi[mask], vi[mask]
        y_dense[i, :, 0] = np.interp(t_dense, ti, xi)
        y_dense[i, :, 1] = np.interp(t_dense, ti, vi)

    return t_dense, y_dense


def dedup_solutions(t_dense, y_dense, rms_tol=1e-4):
    """
    Keep unique solutions by RMS distance on x(t) over the common grid.
    Return indices of unique solutions.
    """
    keep = []
    for i in range(y_dense.shape[0]):
        xi = y_dense[i, :, 0]
        is_new = True
        for k in keep:
            xk = y_dense[k, :, 0]
            rms = np.sqrt(np.mean((xi - xk) ** 2))
            if rms < rms_tol:
                is_new = False
                break
        if is_new:
            keep.append(i)
    return keep


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_overlaid(t_dense, y_dense, idx_keep, title_prefix="Duffing (JAX multistart)"):
    if len(idx_keep) == 0:
        print("No unique solutions to plot.")
        return

    # Time series overlay
    plt.figure(figsize=(10, 4.2))
    for j, i in enumerate(idx_keep, start=1):
        plt.plot(t_dense, y_dense[i, :, 0], lw=1.4, label=f"Sol {j}")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"{title_prefix}: time series")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    # Phase portrait overlay
    plt.figure(figsize=(5.8, 5.2))
    for j, i in enumerate(idx_keep, start=1):
        plt.plot(y_dense[i, :, 0], y_dense[i, :, 1], lw=1.4, label=f"Sol {j}")
    plt.xlabel("x")
    plt.ylabel("v = dx/dt")
    plt.title(f"{title_prefix}: phase portraits")
    plt.legend(fontsize=9)
    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    # ----- Parameters (rad/s) -----
    Q      = 1.0e4
    omega0 = 1.0          # rad/s
    gamma  = 1.0
    omega  = 1.1          # rad/s (drive)
    f      = 1.0

    nT = 1
    T = 2.0 * math.pi / omega * nT

    # Mesh & multistart
    m_nodes    = 600
    num_starts = 50
    amp_range  = (0.2, 3.0)
    n_phase    = 10

    # Solver tolerances
    tol_bvp   = 1e-6
    bc_tol    = 1e-6
    max_iters = 12

    # Build RHS & BC
    fun = make_duffing_fun(Q, omega0, gamma, omega, f)
    bc  = periodic_bc

    # Seeds
    A_lin = float(linear_amplitude(Q, omega0, omega, f))
    print(f"Linear amplitude A_lin ≈ {A_lin:.6g}")

    x_batch, y0_batch = build_seed_batch(
        T, omega, A_lin, m_nodes,
        num_starts=num_starts,
        amp_range=amp_range,
        n_phase=n_phase
    )
    print(f"Batch: x_batch {tuple(x_batch.shape)}, y0_batch {tuple(y0_batch.shape)}")

    # Parallel solve (closure captures fun & bc)
    batched_solver = make_batch_solver(fun, bc, tol=tol_bvp, bc_tol=bc_tol, max_iterations=max_iters)
    xs, ys, stats = batched_solver(x_batch, y0_batch)  # xs:(B,m), ys:(B,m,2), stats:(B,3)

    xs_np    = np.asarray(xs)
    ys_np    = np.asarray(ys)
    stats_np = np.asarray(stats)  # (B, 3) -> [iterations, max_rms_res, max_bc_res]
    iters, rms, bc_res = stats_np[:, 0], stats_np[:, 1], stats_np[:, 2]

    # Success filter
    success = (rms <= tol_bvp) & (np.abs(bc_res) <= bc_tol)
    n_conv  = int(np.sum(success))
    print(f"Converged: {n_conv}/{num_starts}")
    if n_conv == 0:
        print("No solutions converged. Increase m_nodes or relax tolerances / widen amp_range.")
        return

    xs_ok = xs_np[success]
    ys_ok = ys_np[success]

    # Resample to a common dense grid & deduplicate
    t_dense, y_dense = resample_to_uniform(xs_ok, ys_ok, n_dense=3000)
    idx_keep = dedup_solutions(t_dense, y_dense, rms_tol=1e-4)
    print(f"Unique solutions: {len(idx_keep)}")

    # Plot
    plot_overlaid(t_dense, y_dense, idx_keep,
                  title_prefix=f"Duffing (nT={nT}, JAX multistart)")

    # Small summary
    kept_global_idxs = np.where(success)[0][idx_keep]
    for j, gi in enumerate(kept_global_idxs, start=1):
        print(f"  Sol {j:2d} | seed #{gi:2d} | iters={int(iters[gi])} | "
              f"rms={rms[gi]:.2e} | bc={bc_res[gi]:.2e}")

if __name__ == "__main__":
    main()
