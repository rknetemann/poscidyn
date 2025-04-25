import numpy as np
from scipy.integrate import solve_ivp

from normal_form import NormalForm

def steady_state_amp(normal_form: NormalForm,
                     omega_d: float,
                     y0: np.ndarray,
                     t_end: float = 250.0,
                     n_steps: int = 500,
                     discard_frac: float = 0.8) -> float:
    """
    Integrate the system at a given drive frequency and return
    max(|q₁|) over the steady-state portion of the response.
    """
    t_eval = np.linspace(0.0, t_end, n_steps)

    sol = solve_ivp(normal_form.rhs,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=y0,
                    t_eval=t_eval,
                    args=(omega_d,),             # pass ω_d into rhs
                    rtol=1e-5, atol=1e-7,
                    method="RK45")          # swap for "Radau"/"BDF" if stiff

    q1 = sol.y[0]                              # first modal coordinate
    tail = q1[int(discard_frac * len(q1)):]    # discard transients
    return float(np.max(np.abs(tail)))