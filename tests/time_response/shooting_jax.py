# --- requirements ---
# pip install "jax[cpu]" diffrax equinox
# (If you have a GPU with CUDA/cuDNN set up, install the matching jaxlib wheel.)

import time
import math
import jax
import jax.numpy as jnp
from jax import jit
import diffrax as dfx
import equinox as eqx

# Enable float64 for better Newton accuracy
jax.config.update("jax_enable_x64", True)

# ---------------- Parameters ----------------
w0    = 1.0       # natural frequency
Q     = 1000.0    # quality factor
gamma = -1.5      # cubic stiffness
F     = 0.2       # forcing amplitude
w     = 1.0       # forcing frequency
T     = 2.0 * math.pi / w

# pack parameters in a simple tuple for clarity
Params = tuple[float, float, float, float, float, float]
params: Params = (w0, Q, gamma, F, w, T)

# ---------------- Dynamics & Jacobian ----------------
def f(t: float, y: jnp.ndarray, p: Params) -> jnp.ndarray:
    """Duffing RHS: y = [q, v]."""
    w0, Q, gamma, F, w, _T = p
    q, v = y
    return jnp.array([
        v,
        -(w0 / Q) * v - w0**2 * q - gamma * q**3 + F * jnp.cos(w * t)
    ])

def J(t: float, y: jnp.ndarray, p: Params) -> jnp.ndarray:
    """Analytic Jacobian df/dy for STM evolution."""
    w0, Q, gamma, F, w, _T = p
    q, v = y
    return jnp.array([
        [0.0, 1.0],
        [-(w0**2 + 3.0 * gamma * q * q), -(w0 / Q)]
    ])

def aug_rhs(t: float, y_aug: jnp.ndarray, p: Params) -> jnp.ndarray:
    """
    Augmented system for [y; vec(X)], where X is 2x2 STM stacked column-wise.
    dy/dt = f(t, y), dX/dt = A(t, y) @ X
    """
    y = y_aug[:2]
    X = y_aug[2:].reshape(2, 2)
    fy = f(t, y, p)
    A  = J(t, y, p)
    dXdt = A @ X
    return jnp.concatenate([fy, dXdt.reshape(-1)])

# ---------------- Flow over one period (state + STM) ----------------
# Build static solver/config once (keeps jitted cache stable)
_solver = dfx.Dopri5()
_controller = dfx.PIDController(rtol=1e-9, atol=1e-12)
_saveat = dfx.SaveAt(ts=[T])  # grab solution exactly at t = T

@eqx.filter_jit
def flow_and_stm(x0: jnp.ndarray, p: Params):
    """Integrate the augmented system from 0 -> T; return y(T), X(T)."""
    X0 = jnp.eye(2).reshape(-1)
    y0_aug = jnp.concatenate([x0, X0])

    term = dfx.ODETerm(lambda t, y, args: aug_rhs(t, y, p))  # close over params
    sol = dfx.diffeqsolve(
        term,
        _solver,
        t0=0.0,
        t1=T,
        dt0=T/200.0,
        y0=y0_aug,
        saveat=_saveat,
        stepsize_controller=_controller,
        max_steps=100_000,
    )
    yT_aug = sol.ys[0]        # shape (6,)
    yT = yT_aug[:2]
    XT = yT_aug[2:].reshape(2, 2)
    return yT, XT

@eqx.filter_jit
def shooting_residual(x0: jnp.ndarray, p: Params) -> jnp.ndarray:
    """S(x0) = Phi_T(x0) - x0."""
    yT, _ = flow_and_stm(x0, p)
    return yT - x0

# ---------------- Seed via (quasi) harmonic balance ----------------
def harmonic_balance_guess(p: Params) -> jnp.ndarray:
    w0, Q, gamma, F, w, _T = p
    # Linear amplitude
    A_lin = F / math.sqrt((w0**2 - w**2)**2 + (w0*w/Q)**2)

    def g(A):
        Delta = (w0**2 + 0.75*gamma*A*A - w**2)
        return A * math.sqrt(Delta**2 + (w0*w/Q)**2) - F

    A = max(1e-6, A_lin)
    for _ in range(20):
        Delta = (w0**2 + 0.75*gamma*A*A - w**2)
        denom = math.sqrt(Delta**2 + (w0*w/Q)**2)
        dDelta_dA = 1.5 * gamma * A
        dg = denom + A * (Delta * dDelta_dA) / denom
        step = -g(A) / dg
        A_new = max(1e-9, A + step)
        if abs(A_new - A) < 1e-10:
            A = A_new
            break
        A = A_new

    phi = math.atan2((w0*w)/Q, (w0**2 + 0.75*gamma*A*A - w**2))
    x0 = jnp.array([A*math.cos(phi), A*w*math.sin(phi)], dtype=jnp.float64)
    return x0

# ---------------- Newton shooting with backtracking ----------------
def newton_shooting(
    p: Params,
    x0: jnp.ndarray | None = None,
    maxit: int = 15,
    tol: float = 1e-10,
):
    if x0 is None:
        x0 = harmonic_balance_guess(p)

    # Warmup JIT (compile once)
    _ = shooting_residual(x0, p).block_until_ready()
    _ = flow_and_stm(x0, p)[0].block_until_ready()

    for k in range(maxit):
        yT, XT = flow_and_stm(x0, p)
        S = yT - x0
        if jnp.linalg.norm(S, ord=jnp.inf) < tol:
            # Floquet multipliers
            mu = jnp.linalg.eigvals(XT)
            return x0, yT, XT, mu, k

        Jsh = XT - jnp.eye(2)

        # Robust step: solve normal equations with tiny ridge reg
        lam = 1e-8
        JTJ = Jsh.T @ Jsh + lam * jnp.eye(2)
        rhs = -(Jsh.T @ S)
        dx = jnp.linalg.solve(JTJ, rhs)

        # Backtracking line search (host loop calling jitted residual)
        r0 = float(jnp.linalg.norm(S))
        lam_bt = 1.0
        accepted = False
        for _ in range(8):
            x_try = x0 + lam_bt * dx
            r_try = float(jnp.linalg.norm(shooting_residual(x_try, p)))
            if r_try < 0.7 * r0:
                x0 = x_try
                accepted = True
                break
            lam_bt *= 0.5
        if not accepted:
            x0 = x0 + dx  # last resort

    raise RuntimeError("Newton did not converge; try a different seed/params.")

# ---------------- Run once and time many runs ----------------
def run_shooting():
    return newton_shooting(params)

if __name__ == "__main__":
    x0, yT, XT, mu, iters = run_shooting()
    print("Converged in", iters, "Newton steps.")
    print("Periodic initial state x0* =", jnp.asarray(x0))
    print("Residual ||Phi_T(x0*)-x0*|| =", float(jnp.linalg.norm(yT - x0)))
    print("Floquet multipliers:", jnp.asarray(mu))

    # Warmup a second time to ensure compilation is out of the timing loop
    _ = run_shooting()

    n_runs = 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = run_shooting()
    total = time.perf_counter() - t0
    print(f"Total time for {n_runs} runs: {total:.4f} s")
    print(f"Average time per run: {total/n_runs:.6f} s")
