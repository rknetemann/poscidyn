import numpy as np
from numpy.linalg import solve, eigvals, norm
from scipy.integrate import solve_ivp

# Parameters
w0   = 1.0      # natural frequency
Q    = 200.0    # quality factor (weak damping -> large Q)
gamma= 1.0      # cubic stiffness
F    = 0.12     # forcing amplitude
w    = 1.0     # forcing frequency
T    = 2*np.pi/w

def f(t, y):
    q, v = y
    return np.array([v, -(w0/Q)*v - w0**2*q - gamma*q**3 + F*np.cos(w*t)])

def J(t, y):
    q, v = y
    return np.array([[0.0, 1.0],
                     [-(w0**2 + 3*gamma*q*q), -(w0/Q)]])

def aug_rhs(t, y_aug):
    """Augmented RHS for [y; vec(X)], X is 2x2 STM stacked column-wise."""
    y = y_aug[:2]
    X = y_aug[2:].reshape(2,2)
    fy = f(t, y)
    A  = J(t, y)
    dXdt = A @ X
    return np.hstack([fy, dXdt.reshape(-1)])

def flow_and_stm(x0):
    """Integrate state and STM over one period T."""
    X0 = np.eye(2).reshape(-1)
    y0_aug = np.hstack([x0, X0])
    sol = solve_ivp(aug_rhs, (0.0, T), y0_aug, method="RK45",
                    rtol=1e-9, atol=1e-12, max_step=T/200)
    yT = sol.y[:2, -1]
    XT = sol.y[2:, -1].reshape(2,2)
    return yT, XT

def shooting_residual(x0):
    yT, _ = flow_and_stm(x0)
    return yT - x0

def harmonic_balance_guess():
    # Linear-response amplitude as a seed for the nonlinear solve
    A_lin = F/np.sqrt((w0**2 - w**2)**2 + (w0*w/Q)**2)

    # Solve scalar amplitude equation with a damped Newton (clamped positive)
    def g(A):
        Delta = (w0**2 + 0.75*gamma*A*A - w**2)
        return A*np.sqrt(Delta**2 + (w0*w/Q)**2) - F

    A = max(1e-6, A_lin)
    for _ in range(20):
        Delta = (w0**2 + 0.75*gamma*A*A - w**2)
        denom = np.sqrt(Delta**2 + (w0*w/Q)**2)
        # derivative dg/dA
        dDelta_dA = 1.5*gamma*A
        dg = denom + A*(Delta*dDelta_dA)/denom
        step = -g(A)/dg
        A_new = max(1e-9, A + step)
        if abs(A_new - A) < 1e-10:
            A = A_new
            break
        A = A_new

    phi = np.arctan2((w0*w)/Q, (w0**2 + 0.75*gamma*A*A - w**2))
    x0 = np.array([A*np.cos(phi), A*w*np.sin(phi)])
    return x0

def newton_shooting(x0=None, maxit=15, tol=1e-10):
    if x0 is None:
        x0 = harmonic_balance_guess()
        print("Harmonic balance guess: ", x0)

    for k in range(maxit):
        yT, XT = flow_and_stm(x0)
        F = yT - x0
        
        if norm(F, ord=np.inf) < tol:
            # Floquet multipliers from XT
            multipliers = eigvals(XT)
            return x0, yT, XT, multipliers, k
        
        Jsh = XT - np.eye(2)

        # Solve (XT - I) dx = -F with a tiny Tikhonov if near-singular
        try:
            dx = solve(Jsh, -F)
        except np.linalg.LinAlgError:
            lam = 1e-8
            dx = solve(Jsh.T@Jsh + lam*np.eye(2), -Jsh.T@F)

        # Simple backtracking to ensure residual decrease
        lam = 1.0
        r0 = norm(F)
        for _ in range(8):
            x_try = x0 + lam*dx
            r_try = norm(shooting_residual(x_try))
            if r_try < 0.7*r0:
                x0 = x_try
                break
            lam *= 0.5
        else:
            x0 = x0 + dx  # last resort

    raise RuntimeError("Newton did not converge; try a different seed/params.")

# --- run it ---
x0, yT, XT, mu, iters = newton_shooting(np.array([1.0, 0.0]))
print("Converged in", iters, "Newton steps.")
print("Periodic initial state x0* =", x0)
print("Residual ||Phi_T(x0*)-x0*|| =", norm(yT - x0))
print("Floquet multipliers:", mu)
