import numpy as np
from itertools import product
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Core: periodic Duffing via collocation (period = 2π/omega)
# ------------------------------------------------------------
def duffing_periodic_collocation(Q, omega0, gamma, alpha, omega, f,
                                 nT=1, N=400, tol=1e-6,
                                 A_guess=None, phi_guess=0.0,
                                 max_nodes=100000):
    """
    Solve for an nT*T-periodic solution using collocation.
    Allows specifying a cosine initial guess with amplitude and phase.

    Returns: dict with keys:
      'success', 'sol', 't_dense', 'x_dense', 'v_dense', 'T', 'message'
    """
    T = 2 * np.pi / omega

    def fun(t, y):
        x, v = y
        return np.vstack((
            v,
            -(omega0/Q)*v - (omega0**2)*x - gamma*x**3 + alpha*x**5+ f*np.cos(omega*t)
        ))

    def fun_jac(t, y):
        x, v = y
        J = np.zeros((2, 2, y.shape[1]))
        J[0, 1, :] = 1.0
        J[1, 0, :] = -(omega0**2 + 3.0*gamma*x**2 + 5.0*alpha*x**4)
        J[1, 1, :] = -(omega0/Q)
        return J

    def bc(ya, yb):
        return np.array([ya[0]-yb[0], ya[1]-yb[1]])

    def bc_jac(ya, yb):
        A = np.array([[1.0, 0.0],[0.0, 1.0]])
        B = -A
        return A, B

    # Collocation mesh over nT periods
    t_mesh = np.linspace(0.0, nT*T, N)

    # Linear-response amplitude if not provided
    if A_guess is None:
        A_guess = f / np.sqrt((omega0**2 - omega**2)**2 + (omega*omega0/Q)**2)

    # Cosine initial guess with phase
    x0 = A_guess * np.cos(omega*t_mesh + phi_guess)
    v0 = -A_guess * omega * np.sin(omega*t_mesh + phi_guess)
    y_init = np.vstack((x0, v0))

    sol_bvp = solve_bvp(fun, bc, t_mesh, y_init,
                        fun_jac=fun_jac, bc_jac=bc_jac,
                        tol=tol, max_nodes=max_nodes)

    # Dense sample (one fundamental window of what we solved)
    t_dense = np.linspace(0.0, nT*T, 2000)
    if sol_bvp.success:
        x_dense, v_dense = sol_bvp.sol(t_dense)
        msg = "OK"
    else:
        x_dense = np.full_like(t_dense, np.nan)
        v_dense = np.full_like(t_dense, np.nan)
        msg = sol_bvp.message

    return {
        "success": bool(sol_bvp.success),
        "sol": sol_bvp,
        "t_dense": t_dense,
        "x_dense": x_dense,
        "v_dense": v_dense,
        "T": T,
        "nT": nT,
        "message": msg,
        "A_guess": A_guess,
        "phi_guess": phi_guess
    }


# ------------------------------------------------------------
# Multistart driver
# ------------------------------------------------------------
def multistart_periodic_solutions(Q, omega0, gamma, alpha, omega, f,
                                  nT=1,
                                  num_starts=50,
                                  amp_range=(0.2, 3.0),
                                  tol_bvp=1e-6,
                                  N=400,
                                  max_nodes=150000,
                                  rng_seed=0):
    """
    Try many initial guesses (amplitude & phase) and collect unique
    converged nT-periodic solutions.

    amp_range scales the linear amplitude A_lin: A = scale * A_lin,
    where scale ∈ [amp_range[0], amp_range[1]].

    Returns: dict with:
      'unique' : list of solution dicts (see duffing_periodic_collocation return)
      'all'    : list of all attempts (successful or not)
      'A_lin'  : linear-response amplitude used for scaling
    """
    rng = np.random.default_rng(rng_seed)

    # Linear amplitude for reference
    A_lin = f / np.sqrt((omega0**2 - omega**2)**2 + (omega*omega0/Q)**2)

    # Build a grid of amplitudes and phases roughly totaling num_starts guesses
    # Strategy: choose nA amplitudes and nP phases so nA*nP ≈ num_starts
    nP = min(10, num_starts)                      # up to 10 phases around the circle
    nA = max(1, int(np.ceil(num_starts / nP)))    # amplitudes needed
    amp_scales = np.linspace(amp_range[0], amp_range[1], nA)
    phases = np.linspace(0.0, 2*np.pi, nP, endpoint=False)

    # If we over-shoot, cap after num_starts by random subsampling
    all_pairs = list(product(amp_scales, phases))
    if len(all_pairs) > num_starts:
        all_pairs = list(rng.choice(all_pairs, size=num_starts, replace=False))

    attempts = []
    for scale, phi in all_pairs:
        A0 = scale * A_lin
        res = duffing_periodic_collocation(
            Q, omega0, gamma, alpha, omega, f,
            nT=nT, N=N, tol=tol_bvp,
            A_guess=A0, phi_guess=phi,
            max_nodes=max_nodes
        )
        attempts.append(res)

    # Deduplicate converged solutions:
    # Two solutions are "the same" if RMS difference of x(t) over [0, nT*T] is small.
    # (Forced system breaks continuous phase symmetry, so this is OK.)
    uniq = []
    tol_rms = 1e-4  # tighten/loosen depending on your tolerance and problem scaling
    for r in attempts:
        if not r["success"]:
            continue
        x = r["x_dense"]
        is_new = True
        for u in uniq:
            # interpolate onto the shorter t grid if needed (here both have same t_dense length)
            if len(u["t_dense"]) == len(r["t_dense"]) and np.allclose(u["t_dense"], r["t_dense"]):
                dx = x - u["x_dense"]
                rms = np.sqrt(np.mean(dx*dx))
            else:
                # fallback: resample u onto r's t grid
                xu, _ = u["sol"].sol(r["t_dense"])
                dx = x - xu
                rms = np.sqrt(np.mean(dx*dx))
            if rms < tol_rms:
                is_new = False
                break
        if is_new:
            uniq.append(r)

    return {
        "unique": uniq,
        "all": attempts,
        "A_lin": A_lin
    }


# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------
def plot_overlaid_time_and_phase(unique_solutions, title_prefix="Duffing"):
    """
    Overlay x(t) for each unique solution (one window of what was solved, nT*T),
    and overlay phase portraits (x, v).
    """
    if not unique_solutions:
        print("No converged solutions to plot.")
        return

    # --- Time series overlay ---
    plt.figure(figsize=(10, 4.2))
    for i, r in enumerate(unique_solutions, start=1):
        t = r["t_dense"]
        x = r["x_dense"]
        label = f"Sol {i} (nT={r['nT']})"
        plt.plot(t, x, lw=1.4, label=label)
    T = unique_solutions[0]["T"]
    nT = unique_solutions[0]["nT"]
    for k in range(1, nT+1):
        plt.axvline(k*T, linestyle=":", linewidth=0.8)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"{title_prefix}: overlaid time series (nT={nT})")
    plt.legend(loc="best", ncol=2, fontsize=9)
    plt.tight_layout()

    # --- Phase portrait overlay ---
    plt.figure(figsize=(5.8, 5.2))
    for i, r in enumerate(unique_solutions, start=1):
        x = r["x_dense"]
        v = r["v_dense"]
        plt.plot(x, v, lw=1.4, label=f"Sol {i}")
    plt.xlabel("x")
    plt.ylabel("v = dx/dt")
    plt.title(f"{title_prefix}: overlaid phase portraits")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameters (rad/s) ---
    Q      = 10000.0
    omega0 = 1.0        # rad/s
    gamma  = 0.0005
    alpha = 0.00001
    omega  = 1.00009       # rad/s (drive)
    f      = 1.0*omega0**2/Q

    # Try period-1 first; if you’re exploring subharmonics, set nT=2 or 3.
    nT = 1

    # Multistart search
    results = multistart_periodic_solutions(
        Q, omega0, gamma, alpha, omega, f,
        nT=nT,
        num_starts=200,           # requested ~50 starts
        amp_range=(0.1, 2.0),    # scale range around linear amplitude
        tol_bvp=1e-6,
        N=400,
        max_nodes=15000,
        rng_seed=42
    )

    unique = results["unique"]
    print(f"\nLinear amplitude A_lin ≈ {results['A_lin']:.6g}")
    print(f"Attempts: {len(results['all'])}, Converged: {sum(r['success'] for r in results['all'])}, Unique: {len(unique)}")

    # Print brief info about each unique solution
    for i, r in enumerate(unique, start=1):
        x0, v0 = r["sol"].sol(0.0)
        print(f"  Sol {i}: nT={r['nT']}, T={r['T']:.6g}, "
              f"A_guess={r['A_guess']:.3g}, phi_guess={r['phi_guess']:.3g}, "
              f"periodicity check |y(0)-y(T)|≈{np.linalg.norm(r['sol'].sol(r['t_dense'][-1]) - r['sol'].sol(0.0)):.2e}")

    # Overlay plots
    plot_overlaid_time_and_phase(unique, title_prefix="Duffing (multistart)")

    # If you want to *also* look for 2T subharmonics, repeat with nT=2:
    # results_2T = multistart_periodic_solutions(Q, omega0, gamma, omega, f, nT=2, num_starts=50)
    # plot_overlaid_time_and_phase(results_2T["unique"], title_prefix="Duffing (multistart, nT=2)")
