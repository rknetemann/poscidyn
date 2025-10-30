# duffing_periodic_modified_multiple_shooting_jax.py
# --------------------------------------------------
# Compute a T-periodic solution of the forced Duffing oscillator using
# modified multiple shooting with residual minimization in JAX.

from dataclasses import dataclass
import numpy as np  # only for static Gauss nodes; not traced by JAX
import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd, value_and_grad, lax

# Use float64 for better accuracy
jax.config.update("jax_enable_x64", True)

# -----------------------------
# Problem definition & helpers
# -----------------------------

@dataclass
class DuffingParameters:
    omega0: float  # natural frequency
    Q: float       # quality factor
    gamma: float   # cubic stiffness
    f: float       # forcing amplitude
    omega: float   # forcing frequency

def duffing_residual(x, v, a, t, p: DuffingParameters):
    """
    Residual r(t) of the ODE:
        x¨ + (omega0/Q) x˙ + omega0^2 x + gamma x^3 - f cos(omega t) = 0
    """
    return a + (p.omega0 / p.Q) * v + (p.omega0 ** 2) * x + p.gamma * (x ** 3) - p.f * jnp.cos(p.omega * t)

# -----------------------------
# Shifted Legendre basis (on s in [0,1])
# -----------------------------

def legendre_all_on_minus1_1(x, degree: int):
    """
    Return [P0(x), P1(x), ..., P_degree(x)] on [-1,1].
    Uses a stable three-term recurrence with correct initialization.
    """
    # degree is a Python int (static), so plain Python branching is JIT-safe here.
    if degree == 0:
        return jnp.array([1.0], dtype=jnp.float64)

    # allocate output and set P0, P1
    out = jnp.zeros((degree + 1,), dtype=jnp.float64)
    out = out.at[0].set(1.0)     # P0
    out = out.at[1].set(x)       # P1

    if degree == 1:
        return out

    # recurrence: P_{n+1} = ((2n+1)x P_n - n P_{n-1}) / (n+1), for n>=1
    def body(n, carry):
        Pnm1, Pn, out = carry
        Pnp1 = ((2.0 * n + 1.0) * x * Pn - n * Pnm1) / (n + 1.0)
        out = out.at[n + 1].set(Pnp1)
        return (Pn, Pnp1, out)

    Pnm1, Pn = 1.0, x
    Pnm1, Pn, out = lax.fori_loop(1, degree, body, (Pnm1, Pn, out))
    return out

def shifted_legendre_vector(s, degree: int):
    """
    Shifted Legendre basis on s in [0,1]: phi_k(s) = P_k(2s - 1).
    """
    x = 2.0 * s - 1.0
    return legendre_all_on_minus1_1(x, degree)

def basis_vectors_and_derivatives(s, degree: int):
    """
    Returns (phi(s), dphi/ds(s), d2phi/ds2(s)).
    Uses AD on the shifted basis w.r.t. s.
    """
    def phi_fn(u):
        return shifted_legendre_vector(u, degree)
    phi = phi_fn(s)
    dphi_ds = jax.jacfwd(phi_fn)(s)
    d2phi_ds2 = jax.jacfwd(jax.jacfwd(phi_fn))(s)
    return phi, dphi_ds, d2phi_ds2


# Vectorized over nodes
def make_basis_matrices(s_nodes, degree: int):
    """
    Build matrices Phi, dPhi_ds, d2Phi_ds2 evaluated at all nodes s_nodes.
    Each has shape (num_nodes, degree+1).
    """
    vec_fun = vmap(lambda s: basis_vectors_and_derivatives(s, degree), in_axes=(0,))
    phi, dphi, d2phi = vec_fun(s_nodes)  # each (num_nodes, degree+1)
    return phi, dphi, d2phi

# ----------------------------------------
# Gauss-Legendre nodes/weights on [0, 1]
# ----------------------------------------

def gauss_legendre_on_01(order: int):
    """
    Return Gauss-Legendre nodes and weights on [0,1] as JAX arrays.
    Uses NumPy (once) to generate constants; converted to jnp arrays.
    """
    xi, wi = np.polynomial.legendre.leggauss(order)  # nodes/weights on [-1,1]
    s_nodes = (xi + 1.0) / 2.0
    s_weights = wi / 2.0
    return jnp.array(s_nodes, dtype=jnp.float64), jnp.array(s_weights, dtype=jnp.float64)

# ------------------------------------------------
# Objective: residual integral + continuity penalties
# ------------------------------------------------

@dataclass
class SolverHyperParams:
    num_segments: int = 20          # segment count per period
    poly_degree: int = 5            # polynomial degree per segment (>= 2 recommended)
    gauss_order: int = 16           # Gauss-Legendre order for residual integration
    continuity_weight: float = 1e4  # penalty weight for continuity (x, v) across segments
    anchor_weight: float = 1e2      # penalty weight to fix phase at t=0 (x(0), v(0))
    max_iterations: int = 4000      # optimizer iterations
    learning_rate: float = 5e-2     # Adam step size
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    print_every: int = 200          # printing cadence (set 0 to silence)

def build_loss_function(p: DuffingParameters,
                        hp: SolverHyperParams):
    """
    Build a JIT'able loss function J(C) where C has shape (num_segments, poly_degree+1).
    """
    T = 2.0 * jnp.pi / p.omega
    h = T / hp.num_segments

    # Quadrature nodes/weights on s in [0,1] (constants)
    s_nodes, s_weights = gauss_legendre_on_01(hp.gauss_order)
    # Basis matrices on nodes (constants for given degree/order)
    Phi, dPhi_ds, d2Phi_ds2 = make_basis_matrices(s_nodes, hp.poly_degree)

    # Endpoint basis vectors at s=0 and s=1
    phi0, dphi0, _ = basis_vectors_and_derivatives(jnp.array(0.0), hp.poly_degree)
    phi1, dphi1, _ = basis_vectors_and_derivatives(jnp.array(1.0), hp.poly_degree)

    # Per-segment residual integral (vectorized over segments)
    def segment_residual_integral(c_segment, seg_index):
        # x, v, a on quadrature nodes
        x_nodes = Phi @ c_segment                                 # (M,)
        v_nodes = (dPhi_ds @ c_segment) / h                       # (M,)  using dt = h ds
        a_nodes = (d2Phi_ds2 @ c_segment) / (h * h)               # (M,)

        # absolute time on nodes of this segment
        t_nodes = (seg_index + s_nodes) * h                       # (M,)

        r_nodes = duffing_residual(x_nodes, v_nodes, a_nodes, t_nodes, p)
        # Integral over the segment: ∫_0^h r^2 dt = ∫_0^1 r^2(s) * h ds
        return h * jnp.sum(s_weights * (r_nodes ** 2))

    v_segment_residual = vmap(segment_residual_integral, in_axes=(0, 0))

    # Continuity penalty across segment joints (including periodic closure)
    def continuity_penalty(C):
        # x and v at segment starts (s=0) and ends (s=1)
        x_start = C @ phi0                     # (N,)
        v_start = (C @ dphi0) / h              # (N,)
        x_end   = C @ phi1                     # (N,)
        v_end   = (C @ dphi1) / h              # (N,)

        # Enforce end of j equals start of j+1 (roll for periodicity)
        x_jump = x_end - jnp.roll(x_start, shift=-1)
        v_jump = v_end - jnp.roll(v_start, shift=-1)

        return jnp.sum(x_jump * x_jump + v_jump * v_jump)

    # Loss function J(C, x0_anchor, v0_anchor)
    def loss(C, x0_anchor, v0_anchor):
        # Residuals
        seg_indices = jnp.arange(hp.num_segments, dtype=jnp.float64)
        residual_sum = jnp.sum(v_segment_residual(C, seg_indices))

        # Continuity (includes periodic closure)
        cont = continuity_penalty(C)

        # Phase anchor at t=0 (s=0 of first segment)
        x0 = C[0] @ phi0
        v0 = (C[0] @ dphi0) / h
        anchor = (x0 - x0_anchor) ** 2 + (v0 - v0_anchor) ** 2

        total = residual_sum + hp.continuity_weight * cont + hp.anchor_weight * anchor
        return total

    return jit(loss), T, h, (phi0, dphi0, phi1, dphi1), (Phi, dPhi_ds, d2Phi_ds2), (s_nodes, s_weights)

# --------------------------------
# Adam optimizer (pure JAX)
# --------------------------------

@jit
def adam_init(params):
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    return m, v

@jit
def adam_update(params, grads, m, v, t, lr, beta1, beta2, eps):
    m = beta1 * m + (1.0 - beta1) * grads
    v = beta2 * v + (1.0 - beta2) * (grads * grads)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return params, m, v

# --------------------------------
# Initialization helpers
# --------------------------------

def linear_amplitude_guess(p: DuffingParameters):
    """
    Rough amplitude guess ignoring the cubic term (linear forced oscillator).
    """
    denom = jnp.sqrt((p.omega0 ** 2 - p.omega ** 2) ** 2 + (p.omega0 * p.omega / p.Q) ** 2)
    A = p.f / jnp.maximum(denom, 1e-12)
    return float(A)

def initialize_coefficients_from_sinusoid(p: DuffingParameters,
                                          hp: SolverHyperParams,
                                          amplitude: float,
                                          phase: float = 0.0):
    """
    Initialize coefficients by least-squares fit of x(t) ≈ A cos(omega t + phase) on each segment.
    """
    T = 2.0 * jnp.pi / p.omega
    h = T / hp.num_segments

    # Quadrature nodes on [0,1] for LS fit (reuse Gauss nodes)
    s_nodes, _ = gauss_legendre_on_01(max(hp.gauss_order, hp.poly_degree + 3))
    Phi, _, _ = make_basis_matrices(s_nodes, hp.poly_degree)

    # Build coefficients segment by segment by projecting cosine onto basis via LS
    def per_segment_fit(j_idx):
        t_nodes = (j_idx + s_nodes) * h
        x_nodes = amplitude * jnp.cos(p.omega * t_nodes + phase)
        # Solve (Phi c ≈ x_nodes) in LS sense: c = (Phi^T Phi)^(-1) Phi^T x
        # (small (d+1)x(d+1) system; compute explicitly)
        ATA = Phi.T @ Phi
        ATb = Phi.T @ x_nodes
        c = jnp.linalg.solve(ATA, ATb)
        return c

    c0 = vmap(per_segment_fit)(jnp.arange(hp.num_segments))
    return c0  # shape (N, poly_degree+1)

# --------------------------------
# Main solve function
# --------------------------------

def solve_duffing_periodic(p: DuffingParameters,
                           hp: SolverHyperParams,
                           x0_guess: float = None,
                           v0_guess: float = 0.0,
                           init_amplitude: float = None,
                           init_phase: float = 0.0,
                           key: int = 0):
    """
    Solve for a T-periodic solution using modified multiple shooting and residual minimization.

    You can provide either:
      - (x0_guess, v0_guess) to anchor the phase,
      - and/or an initial sinusoid amplitude/phase for coefficient initialization.

    Returns:
      C: coefficients array of shape (num_segments, poly_degree+1)
      T: period
      sample_fn(t_samples) -> (x, v): function to evaluate the solution at arbitrary times in [0, T]
    """
    if init_amplitude is None:
        init_amplitude = linear_amplitude_guess(p)
    if x0_guess is None:
        x0_guess = float(init_amplitude * jnp.cos(init_phase))

    # Build loss
    loss_fn, T, h, endpoints, basis_mats, quad = build_loss_function(p, hp)

    # Initialize coefficients
    C0 = initialize_coefficients_from_sinusoid(p, hp, amplitude=init_amplitude, phase=init_phase)

    # Small random perturbation (optional) to help escape symmetries
    rng = jax.random.PRNGKey(key)
    C0 = C0 + 1e-6 * jax.random.normal(rng, C0.shape, dtype=jnp.float64)

    # Adam optimization
    m, v = adam_init(C0)
    C = C0

    def one_step(carry, i):
        C, m, v = carry
        val, grads = value_and_grad(loss_fn)(C, x0_guess, v0_guess)
        C_new, m_new, v_new = adam_update(
            C, grads, m, v,
            t=i + 1,
            lr=hp.learning_rate,
            beta1=hp.adam_beta1, beta2=hp.adam_beta2, eps=hp.adam_eps
        )
        return (C_new, m_new, v_new), val

    (C_final, _, _), loss_history = lax.scan(one_step, (C, m, v), jnp.arange(hp.max_iterations))

    # Optional: print a few progress values on host
    if hp.print_every and hp.max_iterations >= hp.print_every:
        # materialize to host for printing
        lh = np.asarray(loss_history)
        for k in range(hp.print_every, hp.max_iterations + 1, hp.print_every):
            print(f"[iter {k:5d}] loss = {lh[k-1]:.6e}")

    # Build sampling function: evaluate x(t), v(t)
    phi0, dphi0, phi1, dphi1 = endpoints

    @jit
    def evaluate_segment(C, seg_index, s):
        # basis vectors at arbitrary s
        phi, dphi, _ = basis_vectors_and_derivatives(s, hp.poly_degree)
        x = C[seg_index] @ phi
        v = (C[seg_index] @ dphi) / h
        return x, v

    @jit
    def sample_fn(t_samples):
        """
        Evaluate the periodic solution at times t_samples (JAX array in [0, T]).
        Returns (x, v) arrays of the same shape.
        """
        t_wrap = jnp.mod(t_samples, T)
        seg_pos = t_wrap / h
        seg_idx = jnp.floor(seg_pos).astype(jnp.int32)
        s_local = seg_pos - seg_idx
        xv = vmap(lambda idx, s: evaluate_segment(C_final, idx, s))(seg_idx, s_local)
        return xv[0], xv[1]

    return C_final, float(T), sample_fn

# -----------------------------
# Example usage (adjust params)
# -----------------------------

if __name__ == "__main__":
    # Define Duffing parameters
    params = DuffingParameters(
        omega0=1.0,  # natural frequency
        Q=20.0,      # quality factor (higher -> lower damping)
        gamma=1.0,   # cubic coefficient
        f=0.3,       # forcing amplitude
        omega=0.8    # forcing frequency
    )

    # Solver hyper-parameters
    hparams = SolverHyperParams(
        num_segments=100,
        poly_degree=8,
        gauss_order=30,
        continuity_weight=1e5,  # stronger continuity enforcement
        anchor_weight=1e2,
        max_iterations=30000,
        learning_rate=3e-1,
        print_every=300
    )

    # Initial guess for phase fixation (optional but recommended)
    # Try aligning with linear-response amplitude as a starting point:
    A_lin = linear_amplitude_guess(params)
    x0_guess = float(A_lin)   # anchor x(0) near linear amplitude
    v0_guess = 0.0            # anchor v(0) ~ 0

    # Solve
    C, T, sampler = solve_duffing_periodic(
        p=params,
        hp=hparams,
        x0_guess=x0_guess,
        v0_guess=v0_guess,
        init_amplitude=A_lin,
        init_phase=0.0,
        key=42
    )

    # Sample the solution on a fine grid for inspection (optional)
    import matplotlib.pyplot as plt
    t_plot = jnp.linspace(0.0, T, 2000)
    x_plot, v_plot = sampler(t_plot)

    print(f"Estimated period T = {T:.6f} (target 2π/ω = {2.0*jnp.pi/params.omega:.6f})")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(np.asarray(t_plot), np.asarray(x_plot))
    ax[0].set_ylabel("x(t)")
    ax[0].set_title("Duffing periodic solution (modified multiple shooting, JAX)")

    ax[1].plot(np.asarray(t_plot), np.asarray(v_plot))
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("ẋ(t)")

    plt.tight_layout()
    plt.show()
