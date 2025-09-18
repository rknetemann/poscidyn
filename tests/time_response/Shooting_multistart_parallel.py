# jax_duffing_multiseed_qv.py
import numpy as np
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, lax

jax.config.update("jax_enable_x64", True)

# -----------------------------
# Params as a JAX PyTree
# -----------------------------
@dataclass
class Params:
    w0: float = 1.0
    Q: float  = 200.0
    gamma: float = 1.0
    F: float = 0.12
    w: float = 0.95

def _params_flat(P: Params):
    # children, aux
    return ((jnp.asarray(P.w0),
             jnp.asarray(P.Q),
             jnp.asarray(P.gamma),
             jnp.asarray(P.F),
             jnp.asarray(P.w)), None)

def _params_unflat(aux, children):
    # Do NOT cast to Python floats (tracers break under jit)
    w0, Q, gamma, F, w = children
    return Params(w0, Q, gamma, F, w)

jax.tree_util.register_pytree_node(Params, _params_flat, _params_unflat)

# -----------------------------
# Duffing model & variationals
# -----------------------------
def f(t, y, P: Params):
    q, v = y[0], y[1]
    dq = v
    dv = -(P.w0/P.Q)*v - P.w0**2 * q - P.gamma * q**3 + P.F * jnp.cos(P.w * t)
    return jnp.array([dq, dv])

def J(t, y, P: Params):
    q, _v = y[0], y[1]
    return jnp.array([[0.0, 1.0],
                      [-(P.w0**2 + 3.0*P.gamma*q*q), -(P.w0/P.Q)]])

def f_aug(t, y_aug, P: Params):
    """RHS for [y; vec(X)], X is 2x2 STM stacked column-wise."""
    y  = y_aug[0:2]
    Xv = y_aug[2:]
    X  = Xv.reshape(2,2)
    fy = f(t, y, P)
    A  = J(t, y, P)
    dXdt = A @ X
    return jnp.concatenate([fy, dXdt.reshape(-1)])

# Single RK4 step for augmented system
@jit
def rk4_step(t, h, y_aug, P: Params):
    k1 = f_aug(t,           y_aug,               P)
    k2 = f_aug(t + 0.5*h,   y_aug + 0.5*h*k1,    P)
    k3 = f_aug(t + 0.5*h,   y_aug + 0.5*h*k2,    P)
    k4 = f_aug(t + h,       y_aug + h*k3,        P)
    y_next = y_aug + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return y_next

# Integrate augmented system over one period using lax.scan
@partial(jit, static_argnames=("Nsteps",))
def integrate_augmented_over_period(x0, P: Params, Nsteps: int):
    T = 2.0 * jnp.pi / P.w
    h = T / Nsteps
    y0_aug = jnp.concatenate([x0, jnp.eye(2).reshape(-1)])

    def body_fun(carry, i):
        t = (i.astype(jnp.float64)) * h
        y_next = rk4_step(t, h, carry, P)
        return y_next, None

    yT_aug, _ = lax.scan(body_fun, y0_aug, jnp.arange(Nsteps))
    yT = yT_aug[0:2]
    XT = yT_aug[2:].reshape(2,2)
    return yT, XT

# Batched augmented integration (vmapped)
def v_integrate_augmented(X0, P: Params, Nsteps: int):
    return vmap(lambda x: integrate_augmented_over_period(x, P, Nsteps))(X0)

# Shooting residual for a batch of seeds
@partial(jit, static_argnames=("Nsteps",))
def shooting_residual_batch(X0, P: Params, Nsteps: int):
    yT, _ = v_integrate_augmented(X0, P, Nsteps)
    return yT - X0

# -----------------------------
# Safeguarded batched Newton/LM
# -----------------------------
@partial(jit, static_argnames=("Nsteps","lam_reg","max_backtracks"))
def newton_step_batch(X0, P, Nsteps: int, lam_reg=1e-10, max_backtracks: int = 8):
    """
    One safeguarded Newton/LM step for all seeds:
    - Direct per-seed solve of (XT - I) dx = -S with tiny ridge if needed
    - Backtracking line search that MUST reduce the residual vs r0
    - If no alpha helps, shrink the step (trust region) and retry once
    """
    # integrate to get residual and Jacobian
    yT, XT = v_integrate_augmented(X0, P, Nsteps)   # (B,2), (B,2,2)
    S = yT - X0                                     # (B,2)
    Jshoot = XT - jnp.eye(2)[None, :, :]            # (B,2,2)

    # Direct solve per seed; add tiny ridge if (near-)singular
    def solve_dx(J, b):
        det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
        J_reg = jnp.where(jnp.abs(det) < 1e-12, J + lam_reg*jnp.eye(2), J)
        return jnp.linalg.solve(J_reg, b)

    dx = vmap(lambda J,b: solve_dx(J, -b))(Jshoot, S)  # (B,2)

    # Trust-region clip: prevent huge jumps when far away
    scale = 1.0 + jnp.maximum(jnp.abs(X0[:,0]), jnp.abs(X0[:,1]))[:,None]
    max_step = 0.5 * scale
    dx = jnp.clip(dx, -max_step, max_step)

    # Backtracking candidates
    alphas = jnp.array([1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125], dtype=jnp.float64)

    # Current residual sup-norm
    r0 = jnp.max(jnp.abs(S), axis=1)  # (B,)

    # Evaluate residuals for all alphas and pick the smallest that beats r0
    def eval_alpha(alpha):
        Xtrial = X0 + alpha * dx
        S_trial = shooting_residual_batch(Xtrial, P, Nsteps)
        r_trial = jnp.max(jnp.abs(S_trial), axis=1)
        return r_trial, Xtrial

    r_list, X_list = zip(*[eval_alpha(a) for a in alphas])
    r_mat = jnp.stack(r_list, axis=1)   # (B, n_alpha)
    X_mat = jnp.stack(X_list, axis=1)   # (B, n_alpha, 2)

    # Choose first alpha that gives sufficient decrease
    target = 0.7 * r0[:, None]          # require 30% drop
    ok = r_mat < target                 # (B, n_alpha)

    has_ok = jnp.any(ok, axis=1)        # (B,)
    idx_ok = jnp.argmax(ok, axis=1)     # first True along axis
    idx_min = jnp.argmin(r_mat, axis=1)

    idx = jnp.where(has_ok, idx_ok, idx_min)
    Xcand = jnp.take_along_axis(X_mat, idx[:, None, None], axis=1)[:,0,:]
    rcand = jnp.take_along_axis(r_mat, idx[:, None], axis=1)[:,0]

    # If still not improved, do a single “LM bump”: shrink step
    need_bump = rcand >= r0
    def do_bump(args):
        X0_i, dx_i = args
        dx_i = 0.2 * dx_i  # shrink step
        Xtry = X0_i + dx_i
        Stry = shooting_residual_batch(Xtry[None,:], P, Nsteps)[0]
        rtry = jnp.max(jnp.abs(Stry))
        return jnp.where(rtry < rcand[0], Xtry, X0_i), jnp.minimum(rtry, rcand[0])

    def bump_one(i, carry):
        Xbest, rbest = carry
        X_i = Xbest[i]; r_i = rbest[i]
        dx_i = dx[i]
        X_new, r_new = do_bump((X_i, dx_i))
        Xbest = Xbest.at[i].set(jnp.where(need_bump[i], X_new, X_i))
        rbest = rbest.at[i].set(jnp.where(need_bump[i], r_new, r_i))
        return Xbest, rbest

    Xbest, rbest = Xcand, rcand
    Xbest, rbest = lax.fori_loop(0, X0.shape[0], bump_one, (Xbest, rbest))

    return Xbest, r0, XT

def newton_shooting_batch(X0, P, Nsteps=1200, maxit=25, tol=1e-10, lam_reg=1e-10):
    """
    Safeguarded batched Newton/LM with scan. Higher defaults improve robustness
    for weak damping (XT ~ I).
    """
    X = jnp.array(X0)

    def body_fun(carry, _k):
        Xk = carry
        Xnext, r_prev, XT = newton_step_batch(Xk, P, Nsteps, lam_reg=lam_reg)
        return Xnext, {"r": r_prev, "XT": XT}

    Xk, hist = lax.scan(body_fun, X, jnp.arange(maxit))

    # Final residual after last step:
    S_final = shooting_residual_batch(Xk, P, Nsteps)
    r_final = jnp.max(jnp.abs(S_final), axis=1)

    converged = r_final < tol
    return Xk, r_final, converged, hist

# -----------------------------
# Waveform sampling (batched)
# -----------------------------
@partial(jit, static_argnames=("N",))
def sample_waveform_batch(X0, P: Params, N=256):
    # integrate state (no STM) with RK4; save q at N uniform times
    T = 2.0 * jnp.pi / P.w
    h = T / N

    def rk4_state_step(t, y):
        k1 = f(t, y, P)
        k2 = f(t + 0.5*h, y + 0.5*h*k1, P)
        k3 = f(t + 0.5*h, y + 0.5*h*k2, P)
        k4 = f(t + h,     y + h*k3,     P)
        return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def integrate_one(x0):
        def body(carry, i):
            t = (i.astype(jnp.float64)) * h
            y_next = rk4_state_step(t, carry)
            return y_next, y_next[0]  # keep q only
        _, qs = lax.scan(body, x0, jnp.arange(N))
        return qs

    return vmap(integrate_one)(X0)   # (B, N)

@partial(jit, static_argnames=("Nsteps",))
def floquet_multipliers_batch(X0, P, Nsteps=1200):
    # Integrate augmented system once; return eigenvalues of XT
    _, XT = v_integrate_augmented(X0, P, Nsteps)
    # eigvals per seed (B, 2); jnp.linalg.eigvals works with broadcasting
    mu = vmap(jnp.linalg.eigvals)(XT)   # complex dtype (B, 2)
    return mu

# -----------------------------
# Clustering (CPU / NumPy)
# -----------------------------
def greedy_cluster_by_rms(qs_np, tol_rms=1e-4):
    """
    Greedy de-duplication on CPU (NumPy), qs_np shape (M, N).
    Returns list of clusters, each with indices of members.
    """
    M = qs_np.shape[0]
    used = np.zeros(M, dtype=bool)
    clusters = []
    for i in range(M):
        if used[i]: continue
        rep = qs_np[i]
        group = [i]; used[i] = True
        # vectorized RMS distance
        diff = qs_np - rep[None, :]
        d = np.sqrt(np.mean(diff*diff, axis=1))
        near = np.where((d < tol_rms) & (~used))[0]
        for j in near: used[j] = True
        group.extend(near.tolist())
        clusters.append(group)
    return clusters

# -----------------------------
# Seed construction (manual, using velocity)
# -----------------------------
def seeds_from_rect_grid(q0_list, v0_list):
    """Direct rectangular grid in (q0, v0)."""
    q0 = jnp.asarray(q0_list, dtype=jnp.float64)
    v0 = jnp.asarray(v0_list, dtype=jnp.float64)
    QQ, VV = jnp.meshgrid(q0, v0, indexing="ij")
    return jnp.stack([QQ.reshape(-1), VV.reshape(-1)], axis=1)

def seeds_from_A_v(A_list, vfrac_list, w):
    """
    Velocity-first amplitude seeds.

    Inputs:
      - A_list: amplitudes to target (array-like, A > 0)
      - vfrac_list: velocity fraction α in [-1, 1] so v0 = α * A * w
      - w: forcing frequency (for scaling v0 and the ellipse constraint)

    For each (A, α) we set v0 = α A w and compute q0 = ±sqrt(max(A^2 - (v0/w)^2, 0)).
    Returns both signs of q0 (two seeds per (A, α)).
    """
    A = jnp.asarray(A_list, dtype=jnp.float64)
    a = jnp.asarray(vfrac_list, dtype=jnp.float64)
    AA, AA2 = jnp.meshgrid(A, A, indexing="ij") if False else (None, None)  # silence lints
    AA, AAfrac = jnp.meshgrid(A, a, indexing="ij")  # (NA, Nalpha)

    v0 = AAfrac * AA * w                           # (NA, Nalpha)
    qabs = jnp.sqrt(jnp.maximum(AA*AA - (v0/w)**2, 0.0))  # (NA, Nalpha)

    q0_pos = qabs.reshape(-1)
    q0_neg = -qabs.reshape(-1)
    v0_flat = v0.reshape(-1)

    seeds_pos = jnp.stack([q0_pos, v0_flat], axis=1)
    seeds_neg = jnp.stack([q0_neg, v0_flat], axis=1)
    return jnp.vstack([seeds_pos, seeds_neg])

def add_jitter(seeds, scale_q=0.0, scale_v=0.0, n_per=0, key=jax.random.PRNGKey(0)):
    if n_per <= 0: return seeds
    B = seeds.shape[0]
    dq = scale_q * jax.random.normal(key, (B*n_per,))
    dv = scale_v * jax.random.normal(key, (B*n_per,))
    jittered = seeds.repeat(n_per, axis=0) + jnp.stack([dq, dv], axis=1)
    return jnp.vstack([seeds, jittered])

# -----------------------------
# Example usage (single ω)
# -----------------------------
if __name__ == "__main__":
    P = Params(w0=1.0, Q=200.0, gamma=1.0, F=0.12, w=1.275)
    T_py = 2.0 * np.pi / float(P.w)  # printing only
    print(f"ω={float(P.w):.4f}, T={T_py:.6f}")

    # --- MANUAL SEEDS USING VELOCITY ---
    # (A) Amplitude–velocity seeds: choose amplitudes and velocity fractions α∈[-1,1]
    A_list     = jnp.linspace(0.05, 2.0, 16)           # target amplitudes
    vfrac_list = jnp.linspace(-1.0, 1.0, 13)           # α grid for v0 = α A ω
    seeds_Av   = seeds_from_A_v(A_list, vfrac_list, w=P.w)

    # (B) Optional rectangular grid in (q0, v0)
    q0_list = jnp.linspace(-2.0, 2.0, 17)
    v0_list = jnp.linspace(-3.0, 3.0, 19)
    seeds_rect = seeds_from_rect_grid(q0_list, v0_list)

    # Combine and (optionally) jitter
    seeds = jnp.vstack([seeds_Av, seeds_rect])
    seeds = add_jitter(seeds, scale_q=1e-3, scale_v=1e-3, n_per=0, key=jax.random.PRNGKey(42))
    print("Total seeds:", int(seeds.shape[0]))

    # --- batched Newton shooting ---
    X_star, r_final, converged, hist = newton_shooting_batch(
        seeds, P, Nsteps=1200, maxit=10, tol=1e-10, lam_reg=1e-10
    )

    mu = floquet_multipliers_batch(X_star[converged], P, Nsteps=1200)
    mu_np = np.asarray(mu)
    stable_mask = np.max(np.abs(mu_np), axis=1) < 1.0

    X_star_np = np.asarray(X_star)
    r_np = np.asarray(r_final)
    mask = np.asarray(converged)
    print(f"Converged: {mask.sum()} / {mask.size}")

    # --- sample waveforms for converged solutions (batched) ---
    if mask.any():
        qs = sample_waveform_batch(X_star[converged], P, N=256)   # (M, N)
        qs_np = np.asarray(qs)

        # --- greedy clustering on CPU ---
        clusters = greedy_cluster_by_rms(qs_np, tol_rms=1e-4)
        print(f"Distinct groups at ω={float(P.w):.4f}: {len(clusters)}")

        # report each group's representative (first member)
        for i, grp in enumerate(clusters, 1):
            idx = grp[0]
            q = qs_np[idx]
            A_rep = 0.5*(q.max() - q.min())
            x0_rep = X_star_np[mask][idx]
            mu_rep = mu_np[idx]
            stable = np.max(np.abs(mu_rep)) < 1.0
            print(f"Branch {i}: members={len(grp)}, A≈{A_rep:.5f}, "
                f"||S||_inf≈{r_np[mask][idx]:.2e}, "
                f"μ={mu_rep}, {'stable' if stable else 'unstable'}")
            print("  x0* =", x0_rep)

    else:
        print("No converged periodic solutions at this configuration.")
