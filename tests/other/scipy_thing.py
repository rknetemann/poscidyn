import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import matplotlib.pyplot as plt

# --------------------------
# BVP: periodic Duffing via collocation
# --------------------------
def duffing_periodic_collocation(Q, omega0, gamma, omega, f,
                                 nT=1, N=400, A_guess=None, tol=1e-6):
    """
    Find an nT*T-periodic solution of the forced Duffing ODE using collocation.
    nT=1 -> period-T; nT=2 -> subharmonic (2T), etc.
    Returns (t_dense, x_dense, v_dense, sol_bvp)
    """
    T = 2*np.pi/omega

    def fun(t, y):
        x, v = y
        return np.vstack((
            v,
            -(omega0/Q)*v - (omega0**2)*x - gamma*x**3 + f*np.cos(omega*t)
        ))

    def fun_jac(t, y):
        x, v = y
        J = np.zeros((2, 2, y.shape[1]))
        J[0, 1, :] = 1.0
        J[1, 0, :] = -(omega0**2 + 3.0*gamma*x**2)
        J[1, 1, :] = -(omega0/Q)
        return J

    # Periodic BCs: y(0) = y(T)
    def bc(ya, yb):
        return np.array([ya[0]-yb[0], ya[1]-yb[1]])

    def bc_jac(ya, yb):
        A = np.array([[1.0, 0.0],[0.0, 1.0]])
        B = -A
        return A, B

    # Initial mesh over nT periods
    t_mesh = np.linspace(0.0, nT*T, N)

    # Linear-response amplitude as an initial guess
    if A_guess is None:
        A_guess = f / np.sqrt((omega0**2 - omega**2)**2 + (omega*omega0/Q)**2)

    x0 = A_guess*np.cos(omega*t_mesh)
    v0 = -A_guess*omega*np.sin(omega*t_mesh)
    y_init = np.vstack((x0, v0))

    sol_bvp = solve_bvp(fun, bc, t_mesh, y_init,
                        fun_jac=fun_jac, bc_jac=bc_jac,
                        tol=tol, max_nodes=100000)

    if not sol_bvp.success:
        raise RuntimeError(f"solve_bvp failed: {sol_bvp.message}")

    # Dense output on one (or nT) period
    t_dense = np.linspace(0.0, nT*T, 2000)
    x_dense, v_dense = sol_bvp.sol(t_dense)
    return t_dense, x_dense, v_dense, sol_bvp


# --------------------------
# IVP RHS
# --------------------------
def duffing_ivp_rhs(t, z, Q, omega0, gamma, omega, f):
    x, v = z
    return [v, -(omega0/Q)*v - (omega0**2)*x - gamma*x**3 + f*np.cos(omega*t)]


# --------------------------
# Check periodicity via IVP integration
# --------------------------
def check_periodicity_with_ivp(sol_bvp, Q, omega0, gamma, omega, f,
                               n_periods=5, rtol=1e-9, atol=1e-11):
    """
    Start an IVP at the BVP state y(0), integrate for n_periods*T, and
    compute: (i) Poincaré residuals at kT, (ii) BVP-vs-IVP error on [0, T].
    """
    T = 2*np.pi/omega

    # Initial state from BVP at t=0 (same forcing phase)
    x0, v0 = sol_bvp.sol(0.0)
    z0 = np.array([x0, v0], dtype=float)

    # Integrate IVP
    t0, tf = 0.0, n_periods*T
    ivp = solve_ivp(
        lambda t, z: duffing_ivp_rhs(t, z, Q, omega0, gamma, omega, f),
        (t0, tf), z0, method='Radau', rtol=rtol, atol=atol, dense_output=True
    )
    if not ivp.success:
        raise RuntimeError(f"IVP integration failed: {ivp.message}")

    # Poincaré residuals
    poincare_residuals = []
    for k in range(1, n_periods+1):
        zk = ivp.sol(k*T)
        res = np.linalg.norm(zk - z0, ord=np.inf)
        poincare_residuals.append(res)

    # Compare IVP and BVP on [0, T]
    t_cmp = np.linspace(0.0, T, 4000)
    y_bvp = sol_bvp.sol(t_cmp)  # (2, len)
    y_ivp = ivp.sol(t_cmp)      # (2, len)
    err = np.abs(y_bvp - y_ivp)
    max_err_x = np.max(err[0])
    max_err_v = np.max(err[1])
    rms_err_x = np.sqrt(np.mean(err[0]**2))
    rms_err_v = np.sqrt(np.mean(err[1]**2))

    # Console report
    print(f"Period T = {T:.9g} s")
    print("Poincaré residuals per period (∞-norm of z(kT) - z(0)):")
    for k, r in enumerate(poincare_residuals, start=1):
        print(f"  k={k:2d}: {r:.3e}")
    print(f"Max abs error over [0, T]:  |x|={max_err_x:.3e}, |v|={max_err_v:.3e}")
    print(f"RMS error over [0, T]:      |x|={rms_err_x:.3e}, |v|={rms_err_v:.3e}")

    return {
        "T": T,
        "poincare_residuals": np.array(poincare_residuals),
        "max_err": np.array([max_err_x, max_err_v]),
        "rms_err": np.array([rms_err_x, rms_err_v]),
        "ivp": ivp
    }


# --------------------------
# Plotting
# --------------------------
def plot_duffing_results(sol_bvp, ivp, omega, n_periods=5):
    """
    Plots:
      1) Time series x(t) over n_periods (IVP), with BVP overlay on first period.
      2) Phase portrait (x, v) for IVP, with BVP overlay on first period.
      3) Optional: return times marked.
    """
    T = 2*np.pi/omega

    # IVP dense sampling over n_periods
    t_ivp = np.linspace(0.0, n_periods*T, 6000)
    x_ivp, v_ivp = ivp.sol(t_ivp)

    # BVP sampling over one period
    t_bvp = np.linspace(0.0, T, 2000)
    x_bvp, v_bvp = sol_bvp.sol(t_bvp)

    # --- Time series ---
    plt.figure(figsize=(9, 4))
    plt.plot(t_ivp, x_ivp, label="IVP x(t)")
    plt.plot(t_bvp, x_bvp, linestyle='--', label="BVP x(t) (one period)")
    for k in range(1, n_periods+1):
        plt.axvline(k*T, linestyle=':', linewidth=1)
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title("Duffing: Time series")
    plt.legend()
    plt.tight_layout()

    # --- Phase portrait ---
    plt.figure(figsize=(5.8, 5.2))
    plt.plot(x_ivp, v_ivp, label="IVP trajectory")
    plt.plot(x_bvp, v_bvp, linestyle='--', label="BVP over one period")
    # mark start point
    x0, v0 = sol_bvp.sol(0.0)
    plt.plot([x0], [v0], marker='o', label="Start (t=0)")
    plt.xlabel("x")
    plt.ylabel("v = dx/dt")
    plt.title("Duffing: Phase portrait")
    plt.legend()
    plt.tight_layout()

    plt.show()


# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    Q      = 10000.0
    omega0 = 1.0   # rad/s
    gamma  = 1.0
    omega  = 1.1   # drive frequency
    f      = 1.0

    # 1) Periodic solution (BVP)
    t, x, v, sol_bvp = duffing_periodic_collocation(Q, omega0, gamma, omega, f, nT=1)
    print(sol_bvp)  # SciPy BVP result summary

    # 2) IVP integration from the BVP state to check periodicity
    results = check_periodicity_with_ivp(sol_bvp, Q, omega0, gamma, omega, f,
                                         n_periods=5, rtol=1e-6, atol=1e-11)

    # 3) Plots: time series & phase portrait
    plot_duffing_results(sol_bvp, results["ivp"], omega, n_periods=5)

    # Optional quick boolean for first return
    tol = 1e-7
    first_return_ok = results["poincare_residuals"][0] < tol
    print(f"First return within tol={tol:g}? -> {first_return_ok}")
