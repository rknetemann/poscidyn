import numpy as np
from numpy.linalg import solve, eigvals, norm
from scipy.integrate import solve_ivp

# -----------------------------
# Duffing model & variationals
# -----------------------------
class DuffingParams:
    def __init__(self, w0=1.0, Q=200.0, gamma=1.0, F=0.12, w=0.95):
        self.w0 = w0; self.Q = Q; self.gamma = gamma; self.F = F; self.w = w
    @property
    def T(self): return 2*np.pi/self.w

def f(t, y, P: DuffingParams):
    q, v = y
    return np.array([
        v,
        -(P.w0/P.Q)*v - P.w0**2*q - P.gamma*q**3 + P.F*np.cos(P.w*t)
    ])

def J(t, y, P: DuffingParams):
    q, v = y
    return np.array([
        [0.0, 1.0],
        [-(P.w0**2 + 3*P.gamma*q*q), -(P.w0/P.Q)]
    ])

def aug_rhs(t, y_aug, P: DuffingParams):
    """RHS for [y; vec(X)], X is 2x2 STM stacked column-wise."""
    y = y_aug[:2]
    X = y_aug[2:].reshape(2,2)
    fy = f(t, y, P)
    A  = J(t, y, P)
    dXdt = A @ X
    return np.hstack([fy, dXdt.reshape(-1)])

def flow_and_stm(x0, P: DuffingParams, rtol=1e-9, atol=1e-12):
    """Integrate state and STM over one forcing period."""
    X0 = np.eye(2).reshape(-1)
    y0_aug = np.hstack([x0, X0])
    sol = solve_ivp(
        lambda t, z: aug_rhs(t, z, P), (0.0, P.T), y0_aug,
        method="RK45", rtol=rtol, atol=atol, max_step=P.T/200
    )
    yT = sol.y[:2, -1]
    XT = sol.y[2:, -1].reshape(2,2)
    return yT, XT

def shooting_residual(x0, P: DuffingParams):
    yT, _ = flow_and_stm(x0, P)
    return yT - x0

def newton_shooting(x0, P: DuffingParams, maxit=15, tol=1e-10, verbose=False):
    """Single-shooting Newton for one-periodic solution at fixed forcing period."""
    x = np.array(x0, dtype=float)
    for k in range(maxit):
        yT, XT = flow_and_stm(x, P)
        S = yT - x
        res = norm(S, ord=np.inf)
        if verbose:
            print(f"  iter {k}: ||S||_inf = {res:.3e}")
        if res < tol:
            mu = eigvals(XT)
            return True, {"x0": x, "XT": XT, "res": res, "iters": k, "mu": mu}
        Jsh = XT - np.eye(2)
        # robust linear solve (tiny Tikhonov if near-singular)
        try:
            dx = solve(Jsh, -S)
        except np.linalg.LinAlgError:
            lam = 1e-8
            dx = solve(Jsh.T@Jsh + lam*np.eye(2), -Jsh.T@S)

        # backtracking line search on residual
        lam = 1.0
        r0 = norm(S)
        updated = False
        for _ in range(8):
            x_try = x + lam*dx
            r_try = norm(shooting_residual(x_try, P))
            if r_try < 0.7*r0:
                x = x_try
                updated = True
                break
            lam *= 0.5
        if not updated:
            x = x + dx  # last resort
    return False, {"msg": "Newton did not converge", "x0_last": x}

# -----------------------------
# Waveform sampling & features
# -----------------------------
def sample_waveform(x0, P: DuffingParams, N=256, rtol=1e-9, atol=1e-12):
    """Return times and q(t) over one period from initial state x0."""
    sol = solve_ivp(
        lambda t, y: f(t, y, P), (0.0, P.T), x0,
        t_eval=np.linspace(0, P.T, N, endpoint=False),
        method="RK45", rtol=rtol, atol=atol, max_step=P.T/200
    )
    q = sol.y[0]
    return sol.t, q

def solution_features(q):
    """Basic features for optional use; we primarily cluster by waveform RMS."""
    A = 0.5*(np.max(q) - np.min(q))
    mean_q = float(np.mean(q))
    # Fundamental & 3rd harmonic magnitudes (FFT)
    Q = np.fft.rfft(q)
    mag1 = np.abs(Q[1]) / len(q)
    mag3 = np.abs(Q[3]) / len(q) if len(Q) > 3 else 0.0
    return np.array([A, mean_q, mag1, mag3])

def waveform_rms_distance(q1, q2):
    return np.sqrt(np.mean((q1 - q2)**2))

# -----------------------------
# Clustering (no sklearn)
# -----------------------------
def cluster_solutions_by_waveform(solutions, tol_rms=1e-4):
    """
    Greedy clustering: pick a representative, group any solution whose waveform
    RMS distance to the representative is < tol_rms.
    """
    clusters = []
    used = np.zeros(len(solutions), dtype=bool)

    # Precompute waveforms
    for s in solutions:
        if "t" not in s:
            s["t"], s["q"] = sample_waveform(s["x0"], s["P"], N=256)

    for i, si in enumerate(solutions):
        if used[i]: continue
        rep = si
        group = [i]
        used[i] = True
        for j, sj in enumerate(solutions):
            if used[j]: continue
            d = waveform_rms_distance(rep["q"], sj["q"])
            if d < tol_rms:
                used[j] = True
                group.append(j)
        clusters.append({"rep_index": i, "members": group})
    return clusters

# -----------------------------
# Multi-seed driver (single ω)
# -----------------------------
def multi_seed_shooting_at_frequency(P: DuffingParams, seeds, tol=1e-10,
                                     tol_rms=1e-4, verbose=False):
    """
    seeds: list of initial states [q0, v0].
    Returns:
      clusters: list of clusters; each has 'rep' (solution dict), 'members'
      solutions: list of all converged solutions with waveform & diagnostics
    """
    solutions = []
    for sidx, x0 in enumerate(seeds):
        ok, out = newton_shooting(x0, P, tol=tol, verbose=verbose)
        if not ok:
            if verbose:
                print(f"Seed {sidx}: no convergence.")
            continue
        sol = {
            "x0": out["x0"],
            "XT": out["XT"],
            "mu": out["mu"],
            "res": out["res"],
            "iters": out["iters"],
            "P": P,
        }
        # attach waveform
        t, q = sample_waveform(sol["x0"], P, N=256)
        sol["t"], sol["q"] = t, q
        sol["feat"] = solution_features(q)
        solutions.append(sol)

    if len(solutions) == 0:
        return [], []

    clusters_idx = cluster_solutions_by_waveform(solutions, tol_rms=tol_rms)
    clusters = []
    for c in clusters_idx:
        rep = solutions[c["rep_index"]]
        clusters.append({
            "rep": rep,
            "members": [solutions[k] for k in c["members"]],
        })
    return clusters, solutions

# -----------------------------
# Helpers: manual seed grids
# -----------------------------
def seeds_from_A_phi(A_list, phi_list, w):
    """
    Convert amplitude/phase guesses (relative to forcing) into (q0, v0).
    v0 uses the forcing frequency w as ω_guess.
    """
    seeds = []
    for A in A_list:
        for phi in phi_list:
            q0 = A*np.cos(phi)
            v0 = A*w*np.sin(phi)
            seeds.append(np.array([q0, v0], dtype=float))
    return seeds

def seeds_from_rect_grid(q0_list, v0_list):
    return [np.array([q0, v0], dtype=float) for q0 in q0_list for v0 in v0_list]

def jitter(seeds, scale_q=0.0, scale_v=0.0, n_per=0):
    """Optionally expand each seed with small random perturbations."""
    if n_per <= 0: return seeds
    out = []
    rng = np.random.default_rng(0)
    for s in seeds:
        out.append(s)
        for _ in range(n_per):
            dq = scale_q * rng.standard_normal()
            dv = scale_v * rng.standard_normal()
            out.append(s + np.array([dq, dv]))
    return out

# -----------------------------
# EXAMPLE USAGE (single frequency)
# -----------------------------
if __name__ == "__main__":
    # 1) Set model/forcing parameters
    P = DuffingParams(w0=1.0, Q=200.0, gamma=1.0, F=0.12, w=0.95)
    print(f"Using ω={P.w:.4f}, T={P.T:.6f}")

    # 2) Manually define your initial-guess grids/ranges
    # (a) Amplitude–phase grid
    A_list   = np.linspace(0.02, 1.8, 12)           # you choose the range & density
    phi_list = np.linspace(0.0, 2*np.pi, 8, endpoint=False)

    seeds_ap = seeds_from_A_phi(A_list, phi_list, w=P.w)

    # (b) Direct rectangular grid in (q0, v0) if you prefer
    q0_list = np.linspace(-1.5, 1.5, 7)             # manual ranges
    v0_list = np.linspace(-1.5, 1.5, 7)
    seeds_rect = seeds_from_rect_grid(q0_list, v0_list)

    # (c) Combine and optionally add small jitter
    seeds = seeds_ap + seeds_rect
    seeds = jitter(seeds, scale_q=1e-3, scale_v=1e-3, n_per=1)  # set n_per=0 to disable

    print(f"Total seeds: {len(seeds)}")

    # 3) Run multi-seed shooting and cluster distinct solutions
    clusters, solutions = multi_seed_shooting_at_frequency(
        P, seeds, tol=1e-10, tol_rms=1e-4, verbose=True
    )

    # 4) Report
    print(f"Converged solutions: {len(solutions)}")
    print(f"Distinct groups at ω={P.w:.4f}: {len(clusters)}\n")

    for bi, cl in enumerate(clusters, 1):
        rep = cl["rep"]
        A_rep = 0.5*(np.max(rep["q"]) - np.min(rep["q"]))
        stable = np.all(np.abs(rep["mu"]) < 1.0)  # stability tag
        print(f"Branch {bi}: members={len(cl['members'])}, "
              f"A≈{A_rep:.4f}, "
              f"||S||_inf={rep['res']:.1e}, "
              f"Floquet |μ| max={np.max(np.abs(rep['mu'])):.6f}, "
              f"{'stable' if stable else 'unstable'}")
        print(f"  Representative x0* = {rep['x0']}\n")
