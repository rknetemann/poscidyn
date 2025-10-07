# JAX Gauss–Legendre orthogonal collocation for a forced Duffing oscillator
# Periodic orbit with T = 2*pi/omega; s in {2,3}; dense Newton using JAX autodiff
# Requires: pip install jax jaxlib matplotlib

from jax import config
config.update("jax_platform_name", "gpu")  # or "gpu"/"tpu"
import jax
import jax.numpy as jnp
from jax import jit, vmap, jacfwd
import numpy as np
import matplotlib.pyplot as plt
import optimistix as optx
import jax.profiler

jax.config.update("jax_enable_x64", True)

# ---------------------------
# Problem / discretization parameters (edit me)
# ---------------------------
w0     = 1.0
Q      = 20.0
gamma  = 1.0
fdrive = 1.0
omega  = 1.50
m      = 128
s      = 3              # supports s=2 or s=3 here
newton_maxit = 10
newton_tol   = 1e-8

# ---------------------------
# Gauss nodes on (0,1) and Lagrange basis (values + derivatives)
# ---------------------------
def gauss_nodes_unit_interval(s: int) -> jnp.ndarray:
    """Gauss–Legendre nodes on (0,1) for s in {2,3}."""
    if s == 2:
        c1 = 0.5 - jnp.sqrt(3.0)/6.0
        c2 = 0.5 + jnp.sqrt(3.0)/6.0
        return jnp.array([c1, c2])
    elif s == 3:
        # map from [-1,1] nodes: {-sqrt(3/5), 0, +sqrt(3/5)} via (x+1)/2
        a = jnp.sqrt(3.0/5.0) / 2.0
        return jnp.array([0.5 - a, 0.5, 0.5 + a])
    else:
        raise ValueError("This minimal demo supports s=2 or s=3 only.")

def lagrange_values(nodes, x_eval):
    """
    L[i,j] = L_j(x_eval[i]) for Lagrange basis on 'nodes'.
    Vectorized, no boolean indexing.
    """
    nodes = jnp.asarray(nodes)          # (p,)
    x_eval = jnp.asarray(x_eval)        # (q,)
    p = nodes.shape[0]
    q = x_eval.shape[0]

    # factors[q, j, k] = (x - nodes[k])/(nodes[j] - nodes[k]), with k=j -> 1
    X = x_eval[:, None, None]           # (q,1,1)
    nodes_k = nodes[None, None, :]      # (1,1,p)
    nodes_j = nodes[None, :, None]      # (1,p,1)
    diff = nodes_j - nodes_k            # (1,p,p)

    eye = jnp.eye(p)[None, :, :]        # (1,p,p)
    factors = (X - nodes_k) / jnp.where(eye, 1.0, diff)
    factors = jnp.where(eye, 1.0, factors)  # set k=j factor to 1

    L = jnp.prod(factors, axis=2)       # (q,p)
    return L

def barycentric_diff_matrix(nodes):
    """
    Differentiation matrix D on the node grid itself.
    """
    x = jnp.asarray(nodes)              # (p,)
    p = x.shape[0]
    diff = x[:, None] - x[None, :]      # (p,p)
    eye = jnp.eye(p, dtype=bool)

    # barycentric weights w_j = 1 / prod_{k!=j} (x_j - x_k)
    w = 1.0 / jnp.prod(jnp.where(~eye, diff, 1.0), axis=1)   # (p,)

    # off-diagonal entries
    Wij = (w[None, :] / w[:, None]) / jnp.where(~eye, diff, 1.0)
    Wij = jnp.where(~eye, Wij, 0.0)

    # diagonal entries
    D = Wij.at[jnp.diag_indices(p)].set(-jnp.sum(Wij, axis=1))
    return D  # (p,p)

def lagrange_basis_and_deriv(nodes, x_eval):
    """
    Returns:
      L:  (q,p) values
      dL: (q,p) derivatives w.r.t. x
    For rows where x_eval is exactly a node, we use the barycentric D rows.
    Else, we use L * sum_{k!=j} 1/(x - x_k).
    """
    nodes = jnp.asarray(nodes)      # (p,)
    x_eval = jnp.asarray(x_eval)    # (q,)
    p = nodes.shape[0]
    L = lagrange_values(nodes, x_eval)  # (q,p)

    # general derivative formula away from nodes
    X = x_eval[:, None]             # (q,1)
    inv = 1.0 / (X - nodes[None, :])          # (q,p)
    inv = jnp.where(jnp.isfinite(inv), inv, 0.0)
    S = inv - jnp.diag(jnp.diag(inv @ jnp.ones((p,1))).flatten())  # not correct; we'll build S cleanly

    # Build S = sum_{k!=j} 1/(x - nodes[k]) for each column j:
    # Make matrix inv_full[q,k] and broadcast to (q,p,k), then zero k=j and sum over k.
    inv_full = 1.0 / (X - nodes[None, :])                      # (q,p)
    inv_full = jnp.where(jnp.isfinite(inv_full), inv_full, 0.0)
    eye = jnp.eye(p)[None, :, :]                               # (1,p,p)
    S = jnp.sum(inv_full[:, None, :] * (1.0 - eye), axis=2)    # (q,p)
    dL_general = L * S                                         # (q,p)

    # Where x_eval equals a node, replace with differentiation matrix row.
    D = barycentric_diff_matrix(nodes)                          # (p,p)
    # For each x_eval row, find the closest node index and check exact equality:
    idx = jnp.argmin(jnp.abs(x_eval[:, None] - nodes[None, :]), axis=1)  # (q,)
    is_node = jnp.isclose(x_eval, nodes[idx], atol=1e-15, rtol=0.0)      # (q,)

    # Build rows: if is_node[i], use one-hot L and D[idx[i],:]; else keep general
    L_node_rows = jax.nn.one_hot(idx, p, dtype=L.dtype)         # (q,p)
    dL_node_rows = D[idx, :]                                    # (q,p)

    L_final  = jnp.where(is_node[:, None], L_node_rows,  L)
    dL_final = jnp.where(is_node[:, None], dL_node_rows, dL_general)
    return L_final, dL_final

# ---------------------------
# Mesh and precomputations
# ---------------------------
T = 2.0 * jnp.pi / omega
tau_nodes = jnp.linspace(0.0, 1.0, m + 1)
h = jnp.diff(tau_nodes)

c = gauss_nodes_unit_interval(s)                 # (s,)
xi_nodes  = jnp.concatenate([jnp.array([0.0]), c])  # (s+1,)
xi_colloc = c
xi_right  = jnp.array([1.0])

L_colloc, dL_colloc = lagrange_basis_and_deriv(xi_nodes, xi_colloc)  # (s, s+1)
L_right,  dL_right  = lagrange_basis_and_deriv(xi_nodes, xi_right)   # (1, s+1)


# ---------------------------
# Vector field du/dtau = T F(T*tau, u)
# ---------------------------
def F_t_u(tt, u):
    x, v = u
    dxdt = v
    dvdt = -(w0/Q)*v - (w0**2)*x - gamma*(x**3) + fdrive*jnp.cos(omega*tt)
    return jnp.array([dxdt, dvdt])

def G_tau_u(tau, u):
    return T * F_t_u(T * tau, u)

# ---------------------------
# Initial guess: linear forced response (gamma=0)
# ---------------------------
def linear_guess_t(t):
    den = (w0**2 - omega**2)**2 + (w0*omega/Q)**2
    A = fdrive / jnp.sqrt(den)
    phi = jnp.arctan2((w0*omega/Q), (w0**2 - omega**2))
    x = A * jnp.cos(omega*t - phi)
    v = -A * omega * jnp.sin(omega*t - phi)
    return jnp.stack([x, v])

@jit
def build_initial_guess():
    # U has shape (m, s+1, 2): per element nodes {0, c1, ..., cs}
    def element_init(k):
        tau_L = tau_nodes[k]
        tau_loc = tau_L + h[k] * xi_nodes       # (s+1,)
        t_loc = T * tau_loc
        U_k = vmap(linear_guess_t)(t_loc).T     # (s+1, 2)
        return U_k
    return vmap(element_init)(jnp.arange(m))

# Unknowns U flattened to vector z and reshape helpers
def U_to_z(U):
    return U.reshape(-1)

def z_to_U(z):
    return z.reshape((m, s+1, 2))

U0 = build_initial_guess()
z0 = U_to_z(U0)

# ---------------------------
# Element evaluations (jit + vmap)
# ---------------------------
@jit
def element_collocation_values(Uk):
    # (s, s+1) @ (s+1, 2) -> (s,2)
    return L_colloc @ Uk

@jit
def element_collocation_derivs(Uk, hk):
    # d/dtau = (1/hk) * d/dxi
    return (dL_colloc @ Uk) / hk  # (s,2)

@jit
def element_right(Uk):
    return (L_right @ Uk)[0]      # (2,)

# ---------------------------
# Global residual (vectorized over elements/points)
# ---------------------------
def residual(z, args=None):
    U = z_to_U(z)  # (m, s+1, 2)

    # Collocation residuals for each element k and each colloc point i
    def colloc_for_elem(k, Uk):
        hk = h[k]
        tau_L = tau_nodes[k]
        taus_k = tau_L + hk * xi_colloc                      # (s,)
        Uc = element_collocation_values(Uk)                  # (s,2)
        dUc_dtau = element_collocation_derivs(Uk, hk)        # (s,2)
        # residual[i] = dUc_dtau[i] - G(taus_k[i], Uc[i])
        Ri = dUc_dtau - vmap(G_tau_u)(taus_k, Uc)            # (s,2)
        return Ri.reshape(-1)                                 # (2s,)

    R_colloc = vmap(colloc_for_elem, in_axes=(0, 0))(jnp.arange(m), U)  # (m, 2s)
    R_colloc = R_colloc.reshape(-1)

    # Continuity (right_k - left_{k+1})
    right_all = vmap(element_right)(U)             # (m,2)
    left_all  = U[:, 0, :]                         # (m,2)
    cont_pairs = right_all[:-1] - left_all[1:]     # (m-1,2)
    R_cont = cont_pairs.reshape(-1)

    # Periodicity (last right - first left)
    R_per = (right_all[-1] - left_all[0]).reshape(-1)  # (2,)

    return jnp.concatenate([R_colloc, R_cont, R_per])  # length N


solver = optx.LevenbergMarquardt(rtol=1e-4, atol=1e-6, norm=optx.rms_norm) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
sol = optx.least_squares(residual, solver, y0=z0, options={"jac": "bwd"},  throw=False, max_steps=newton_maxit) 

if sol.result == optx.RESULTS.successful:
    print(f"Newton converged")

z_star = sol.value
U_star = z_to_U(z_star)

# ---------------------------
# Reconstruct solution on a fine grid for plotting / stats
# ---------------------------
def lagrange_values_only(nodes, x_eval):
    L, _ = lagrange_basis_and_deriv(nodes, x_eval)
    return L

xi_plot = jnp.linspace(0.0, 1.0, 40)
L_plot = lagrange_values_only(xi_nodes, xi_plot)           # (np, s+1)

# ============================
# Multi-seed search utilities
# ============================

# Linear amplitude/phase helpers (same as your linear_guess_t but returning A,phi)
def linear_amp_phase():
    den = (w0**2 - omega**2)**2 + (w0*omega/Q)**2
    A = fdrive / jnp.sqrt(den)
    phi = jnp.arctan2((w0*omega/Q), (w0**2 - omega**2))
    return A, phi

# Build a seed U (m, s+1, 2) by scaling linear solution and adding a phase shift
def build_seed(scale: float, phase_shift: float):
    A, phi = linear_amp_phase()
    # time locations at element nodes
    def elem_nodes(k):
        tau_L = tau_nodes[k]
        tau_loc = tau_L + h[k] * xi_nodes   # (s+1,)
        t_loc = T * tau_loc + phase_shift / omega  # shift in time = phase/omega
        # linear solution with *unit* amplitude A (already included)
        x = A * jnp.cos(omega*t_loc - phi)
        v = -A * omega * jnp.sin(omega*t_loc - phi)
        # scale both x and v (consistent for linear response)
        x *= scale
        v *= scale
        return jnp.stack([x, v], axis=-1)   # (s+1,2)
    U_seed = vmap(elem_nodes)(jnp.arange(m))
    return U_seed

# One solve from a given seed; returns (success, z, U, stats_dict)
def solve_from_seed(U_seed):
    z_init = U_to_z(U_seed)
    try:
        sol = optx.least_squares(residual, solver, y0=z_init,
                                 options={"jac": "bwd"}, throw=False,
                                 max_steps=newton_maxit)
        ok = (sol.result == optx.RESULTS.successful)
        z_sol = sol.value
    except Exception:
        ok = False
        z_sol = z_init

    if not ok:
        return False, z_sol, None, None

    U_sol = z_to_U(z_sol)

    # Reconstruct for amplitude stats (reuse your L_plot machinery)
    def sample_elem(k, Uk):
        tau_L = tau_nodes[k]
        tau_seg = tau_L + h[k] * xi_plot
        vals = L_plot @ Uk
        return tau_seg, vals
    taus_list, vals_list = vmap(sample_elem, in_axes=(0, 0))(jnp.arange(m), U_sol)
    x_all = vals_list.reshape(-1, 2)[:, 0]
    amp_peak = jnp.max(jnp.abs(x_all))
    x0, v0 = U_sol[0, 0, 0], U_sol[0, 0, 1]
    stats = {"amp_peak": float(amp_peak), "x0": float(x0), "v0": float(v0)}
    return True, z_sol, U_sol, stats

# Simple deduplication: compare by |amp_peak| and x(0), v(0)
def is_duplicate(stats_a, stats_b, tol_amp=1e-3, tol_ic=1e-3):
    if abs(stats_a["amp_peak"] - stats_b["amp_peak"]) > tol_amp:
        return False
    if abs(stats_a["x0"] - stats_b["x0"]) > tol_ic:
        return False
    if abs(stats_a["v0"] - stats_b["v0"]) > tol_ic:
        return False
    return True

def insert_unique(solutions, entry, tol_amp=1e-3, tol_ic=1e-3):
    for e in solutions:
        if is_duplicate(e["stats"], entry["stats"], tol_amp, tol_ic):
            return False
    solutions.append(entry)
    return True

# ----------------------------------------
# Build grid of seeds and run all solves
# ----------------------------------------
# Base factor and grid — tweak as you like:
BASE_FACTOR = 1.0
SCALE_FACTORS = BASE_FACTOR * jnp.linspace(1.0, 2.0, 100)  # Denser grid of scale factors
PHASES = jnp.linspace(0.0, 2*jnp.pi, 10, endpoint=False)   # Denser grid of phase shifts

solutions = []
n_tried = 0
for sf in np.asarray(SCALE_FACTORS):
    for ph in np.asarray(PHASES):
        U_seed = build_seed(float(sf), float(ph))
        ok, z_sol, U_sol, stats = solve_from_seed(U_seed)
        n_tried += 1
        if not ok:
            print(f"  Seed scale={sf:.3f}, phase={ph:.3f} rad  --> no convergence")
            continue
        entry = {"z": z_sol, "U": U_sol, "stats": stats, "scale": float(sf), "phase": float(ph)}
        print(f"  Found solution: amp_peak={stats['amp_peak']:.6f}  x0={stats['x0']:.6f}  v0={stats['v0']:.6f}   (seed scale={sf:.3f}, phase={ph:.3f} rad)")
        insert_unique(solutions, entry, tol_amp=1e-3, tol_ic=1e-3)

# Sort solutions by amplitude descending
solutions.sort(key=lambda e: e["stats"]["amp_peak"], reverse=True)

# ---------------------------
# Summary printout
# ---------------------------
print("\n=== Multi-seed search summary ===")
print(f"Tried seeds: {n_tried}, converged unique solutions: {len(solutions)}")
for i, e in enumerate(solutions, 1):
    st = e["stats"]
    print(f"[{i}] amp_max={st['amp_peak']:.6f}  x0={st['x0']:.6f}  v0={st['v0']:.6f}   "
          f"(seed scale={e['scale']:.3f}, phase={e['phase']:.3f} rad)")

# Optional: pick one to plot (largest amplitude)
if len(solutions) > 0:
    U_pick = solutions[0]["U"]
    # sample for plotting
    def sample_elem(k, Uk):
        tau_L = tau_nodes[k]; tau_seg = tau_L + h[k] * xi_plot
        vals = L_plot @ Uk
        return tau_seg, vals
    taus_list, vals_list = vmap(sample_elem, in_axes=(0, 0))(jnp.arange(m), U_pick)
    t_all = (T * taus_list.reshape(-1))
    x_all = (vals_list.reshape(-1, 2)[:, 0])
    order = jnp.argsort(t_all)
    t_all = np.asarray(t_all[order])
    x_all = np.asarray(x_all[order])

    plt.figure(figsize=(8,4))
    plt.plot(t_all, x_all)
    plt.xlabel("t"); plt.ylabel("x(t)")
    plt.title(f"Picked solution #{1} (max |x|={solutions[0]['stats']['amp_peak']:.4f})")
    plt.tight_layout()
    # plt.show()

# build per-element fine samples
def sample_elem(k, Uk):
    tau_L = tau_nodes[k]
    tau_seg = tau_L + h[k] * xi_plot                       # (np,)
    vals = L_plot @ Uk                                     # (np,2)
    return tau_seg, vals

taus_list, vals_list = vmap(sample_elem, in_axes=(0, 0))(jnp.arange(m), U_star)
taus_all = taus_list.reshape(-1)
vals_all = vals_list.reshape(-1, 2)
t_all = T * taus_all
x_all = vals_all[:, 0]

# sort for neat plot
order = jnp.argsort(t_all)
t_all = t_all[order]
x_all = x_all[order]

# Amplitude stats
imax = jnp.argmax(x_all)
imin = jnp.argmin(x_all)
x_max = x_all[imax]
x_min = x_all[imin]
t_at_xmax = t_all[imax]
t_at_xmin = t_all[imin]
amp_peak = jnp.max(jnp.abs(x_all))
amp_half_pp = 0.5 * (x_max - x_min)

print(f"Endpoint check (interp): x(0)={float(U_star[0,0,0]):.6f}, vs x(T-end)={float((L_right @ U_star[-1])[0][0]):.6f}")
print(f"Max x over period: {float(x_max):.6f} at t={float(t_at_xmax):.6f}")
print(f"Min x over period: {float(x_min):.6f} at t={float(t_at_xmin):.6f}")
print(f"Peak amplitude (max |x|): {float(amp_peak):.6f}")
print(f"Half peak-to-peak amplitude: {float(amp_half_pp):.6f}")

# Plot
plt.figure(figsize=(8,4))
plt.plot(np.asarray(t_all), np.asarray(x_all))
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title(f"Duffing periodic orbit via Gauss collocation (s={s}, m={m}) [JAX]")
plt.tight_layout()
# plt.show()

jax.profiler.save_device_memory_profile("memory.prof")