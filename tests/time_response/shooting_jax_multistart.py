# --- requirements ---
# pip install "jax[cpu]" diffrax equinox

import math
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
from typing import NamedTuple

jax.config.update("jax_enable_x64", True)

# ---------------- Parameters ----------------
w0    = 1.0
Q     = 1000.0
gamma = -1.5
F     = 0.2
w     = 1.0
T     = 2.0 * math.pi / w

Params = tuple[float, float, float, float, float, float]
params: Params = (w0, Q, gamma, F, w, T)

# ---------------- Dynamics & Jacobian ----------------
def f(t: float, y: jnp.ndarray, p: Params) -> jnp.ndarray:
    w0, Q, gamma, F, w, _T = p
    q, v = y
    return jnp.array([
        v,
        -(w0 / Q) * v - w0**2 * q - gamma * q**3 + F * jnp.cos(w * t)
    ])

def J(t: float, y: jnp.ndarray, p: Params) -> jnp.ndarray:
    w0, Q, gamma, F, w, _T = p
    q, v = y
    return jnp.array([
        [0.0, 1.0],
        [-(w0**2 + 3.0 * gamma * q * q), -(w0 / Q)]
    ])

def aug_rhs(t: float, y_aug: jnp.ndarray, p: Params) -> jnp.ndarray:
    y = y_aug[:2]
    X = y_aug[2:].reshape(2, 2)
    fy = f(t, y, p)
    A  = J(t, y, p)
    dXdt = A @ X
    return jnp.concatenate([fy, dXdt.reshape(-1)])

# ---------------- One-period flow (y, STM) ----------------
_solver = dfx.Dopri5()
_controller = dfx.PIDController(rtol=1e-9, atol=1e-12)
_saveat = dfx.SaveAt(ts=[T])

@eqx.filter_jit
def flow_and_stm(x0: jnp.ndarray, p: Params):
    X0 = jnp.eye(2).reshape(-1)
    y0_aug = jnp.concatenate([x0, X0])

    term = dfx.ODETerm(lambda t, y, _args: aug_rhs(t, y, p))
    sol = dfx.diffeqsolve(
        term,
        _solver,
        t0=0.0,
        t1=T,
        dt0=T/200.0,
        y0=y0_aug,
        saveat=_saveat,
        stepsize_controller=_controller,
        max_steps=100_000_000,
    )
    yT_aug = sol.ys[0]
    yT = yT_aug[:2]
    XT = yT_aug[2:].reshape(2, 2)
    return yT, XT

@eqx.filter_jit
def shooting_residual(x0: jnp.ndarray, p: Params) -> jnp.ndarray:
    yT, _ = flow_and_stm(x0, p)
    return yT - x0

# ---------------- Pure-JAX Newton with backtracking ----------------
class NewtonResult(NamedTuple):
    x0_star: jnp.ndarray      # converged initial state (or last iterate)
    yT: jnp.ndarray           # Phi_T(x0_star)
    XT: jnp.ndarray           # STM at T
    mu: jnp.ndarray           # Floquet multipliers (Nan if not converged)
    res_norm: float           # ||Phi_T(x)-x||
    iters: int
    success: bool

def _newton_step(x: jnp.ndarray, p: Params):
    """Compute residual, Jacobian, Newton step dx, and (yT, XT)."""
    yT, XT = flow_and_stm(x, p)
    S = yT - x
    Jsh = XT - jnp.eye(2)
    # robust normal equations with small ridge
    lam = 1e-8
    JTJ = Jsh.T @ Jsh + lam * jnp.eye(2)
    rhs = -(Jsh.T @ S)
    dx = jnp.linalg.solve(JTJ, rhs)
    return S, dx, yT, XT

def _backtrack(x: jnp.ndarray, dx: jnp.ndarray, r0: float, p: Params):
    """Try up to 8 halvings to reduce residual by factor 0.7."""
    def body_fun(carry):
        k, lam_bt, x_best, accepted = carry
        x_try = x + lam_bt * dx
        r_try = jnp.linalg.norm(shooting_residual(x_try, p))
        improve = r_try < 0.7 * r0
        x_new = jnp.where(improve, x_try, x_best)
        acc_new = accepted | improve
        lam_new = jnp.where(improve, lam_bt, lam_bt * 0.5)
        return (k + 1, lam_new, x_new, acc_new)

    init = (0, 1.0, x, False)
    cond = lambda c: (c[0] < 8) & (~c[3])
    k, lam_bt, x_acc, accepted = jax.lax.while_loop(cond, body_fun, init)
    # If never accepted, fall back to full step
    x_next = jnp.where(accepted, x_acc, x + dx)
    return x_next

@eqx.filter_jit
def newton_shooting_jax(x0: jnp.ndarray, p: Params, maxit: int = 15, tol: float = 1e-10) -> NewtonResult:
    _ = shooting_residual(x0, p)

    def body_fun(state):
        k, x, done, last_yT, last_XT = state

        def do_step(x):
            S, dx, yT, XT = _newton_step(x, p)
            r0 = jnp.linalg.norm(S)
            x_next = _backtrack(x, dx, r0, p)
            S_next = shooting_residual(x_next, p)
            done_now = jnp.linalg.norm(S_next, ord=jnp.inf) < tol
            return x_next, yT, XT, done_now, S_next

        x_new, yT, XT, done_now, S_next = jax.lax.cond(
            done,
            lambda *_: (x, last_yT, last_XT, True, shooting_residual(x, p)),
            lambda *_: do_step(x),
            operand=None
        )
        return (k + 1, x_new, done | done_now, yT, XT)

    init_state = (0, x0, False, jnp.zeros(2, dtype=jnp.float64), jnp.eye(2, dtype=jnp.float64))

    def cond_fun(s):
        k, x, done, _, _ = s
        return (k < maxit) & (~done)

    k_final, x_star, done_final, yT_fin, XT_fin = jax.lax.while_loop(cond_fun, body_fun, init_state)

    res = shooting_residual(x_star, p)
    resn = jnp.linalg.norm(res)

    # --- FIX: make both branches complex128 ---
    mu = jax.lax.cond(
        done_final,
        lambda XT: jnp.asarray(jnp.linalg.eigvals(XT), dtype=jnp.complex128),
        lambda XT: jnp.full((2,), jnp.nan + 0j, dtype=jnp.complex128),
        XT_fin
    )

    # Keep success as a JAX boolean; cast outside jit if you want a Python bool
    success = done_final

    return NewtonResult(
        x0_star=x_star,
        yT=yT_fin,
        XT=XT_fin,
        mu=mu,
        res_norm=resn,
        iters=k_final,
        success=success,
    )

# ---------------- Batch / Parallel solve over a grid ----------------
def make_ic_grid(qmin=-2.0, qmax=2.0, vmin=-2.0, vmax=2.0, nq=21, nv=21) -> jnp.ndarray:
    q = jnp.linspace(qmin, qmax, nq)
    v = jnp.linspace(vmin, vmax, nv)
    Qg, Vg = jnp.meshgrid(q, v, indexing="xy")
    grid = jnp.stack([Qg.ravel(), Vg.ravel()], axis=-1)  # (nq*nv, 2)
    return grid

# vmapped parallel Newton
vmapped_newton = eqx.filter_jit(jax.vmap(lambda x0: newton_shooting_jax(x0, params)))

# ---------------- Deduplicate converged solutions (host side) -------
def dedupe_solutions(x_stars: jnp.ndarray, success: jnp.ndarray, tol: float = 1e-6):
    """
    Greedy host-side deduplication by Euclidean distance.
    Returns indices of representatives and an assignment array.
    """
    import numpy as np
    X = np.array(x_stars)[np.array(success)]
    if X.size == 0:
        return [], np.array([], dtype=int)

    reps = []
    assign = -np.ones(X.shape[0], dtype=int)
    for i, xi in enumerate(X):
        if i == 0:
            reps.append(0)
            assign[i] = 0
            continue
        d2 = ((X[reps] - xi) ** 2).sum(axis=1)
        j = np.argmin(d2)
        if d2[j] > tol**2:
            reps.append(i)
            assign[i] = len(reps) - 1
        else:
            assign[i] = j
    # map representative positions back to original indices among successes
    success_idx = np.nonzero(np.array(success))[0]
    rep_global_idx = success_idx[reps].tolist()
    return rep_global_idx, assign

# ---------------- Example usage ----------------
if __name__ == "__main__":
    # Build a grid of initial guesses
    # (tune ranges/resolution as you like)
    grid = make_ic_grid(qmin=-2.0, qmax=2.0, vmin=-2.0, vmax=2.0, nq=31, nv=31)  # shape (961, 2)

    # Run all solves in parallel (JIT-compiled)
    results: NewtonResult = vmapped_newton(grid)

    # Unpack batched results
    x0_stars = results.x0_star       # (N, 2)
    yTs      = results.yT            # (N, 2)
    XTs      = results.XT            # (N, 2, 2)
    mus      = results.mu            # (N, 2)
    resn     = results.res_norm      # (N,)
    iters    = results.iters         # (N,)
    success  = jnp.array(results.success)  # (N,)

    # Report counts
    n_total = grid.shape[0]
    n_ok = int(success.sum())
    print(f"Solved {n_ok}/{n_total} initial guesses (tol = 1e-10).")

    # Optional: deduplicate converged solutions to representative branches
    reps, assign = dedupe_solutions(x0_stars, success, tol=1e-6)
    print(f"Unique solutions found (within tol): {len(reps)}")
    for k, idx in enumerate(reps):
        xk = jnp.asarray(x0_stars[idx])
        muk = jnp.asarray(mus[idx])
        rk = float(resn[idx])
        print(f"[{k}] x0* = {xk},  ||S|| = {rk:.3e},  mu = {muk}")
