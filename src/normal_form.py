import jax.numpy as jnp
from typing import Sequence, Dict, Callable
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import time

def normal_form(parameters_modes: Sequence[Dict[str, jnp.ndarray]],
                drive_frequency: float
               ) -> Callable[[float, jnp.ndarray], jnp.ndarray]:
    """
    Returns RHS(t, y) for the 1st-order system written in the image.
    
    y = [q₁, …, q_N, q̇₁, …, q̇_N]  (length 2N)

    Each entry in `parameters_modes` must be a dictionary that contains
    (for its own mode j):
        'zeta'  : scalar damping ratio ζ_j
        'omega' : scalar natural freq  ω_j
        'z1'    : (N,)      array        z_{j i}
        'z2'    : (N,N)     array        z_{j i k}
        'z3'    : (N,N,N)   array        z_{j i k l}
        'f'     : scalar forcing amplitude f_j
    """
    N = len(parameters_modes)
    
    if N < 1:
        raise ValueError("parameters_modes must contain at least one mode")

    zeta  = np.array([p["zeta"]  for p in parameters_modes])      # (N,)
    omega = np.array([p["omega"] for p in parameters_modes])      # (N,)
    f_vec = np.array([p["f"]     for p in parameters_modes])      # (N,)

    z1 = np.stack([p["z1"] for p in parameters_modes])            # (N, N)
    z2 = np.stack([p["z2"] for p in parameters_modes])            # (N, N, N)
    z3 = np.stack([p["z3"] for p in parameters_modes])            # (N, N, N, N)]
    
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        q  = y[:N]          # modal coordinates
        qd = y[N:]          # modal velocities

        lin_damp   = -2.0 * zeta * omega * qd
        lin_stiff  = -np.einsum("ji,i->j",   z1, q)
        quad_stiff = -np.einsum("jik,i,k->j", z2, q, q)
        cub_stiff  = -np.einsum("jikl,i,k,l->j", z3, q, q, q)
        force      =  f_vec * np.cos(drive_frequency * t)

        qdd = lin_damp + lin_stiff + quad_stiff + cub_stiff + force
        return np.concatenate([qd, qdd])

    return rhs

def steady_state_amp(parameters_modes: Sequence[Dict[str, np.ndarray]],
                     drive_frequency: float,
                     y0: np.ndarray | None = None,
                     t_end: float = 250.0,
                     n_steps: int = 500,
                     discard_frac: float = 0.8) -> float:
    """
    Integrate the system at a given drive frequency and return
    max(|q₁|) over the steady-state portion of the response.
    """
    N  = len(parameters_modes)
    if y0 is None:
        y0 = np.zeros(2 * N)

    rhs = normal_form(parameters_modes, drive_frequency)
    t_eval = np.linspace(0.0, t_end, n_steps)

    sol = solve_ivp(rhs,
                    t_span=(t_eval[0], t_eval[-1]),
                    y0=y0,
                    t_eval=t_eval,
                    rtol=1e-5, atol=1e-7,
                    method="RK45")          # swap for "Radau"/"BDF" if stiff

    q1 = sol.y[0]                              # first modal coordinate
    tail = q1[int(discard_frac * len(q1)):]    # discard transients
    return float(np.max(np.abs(tail)))

if __name__ == "__main__":
    # ------ define two coupled modes ------------------------------------
    params = [
        dict(zeta=0.01, omega=5.0,
             z1=np.array([10.0, 1.0]),
             z2=np.array([[0.0, 0.5],
                          [0.5, 0.0]]),
             z3=np.zeros((2, 2, 2)),
             f =1.0),
        dict(zeta=0.02, omega=8.0,
             z1=np.array([1.0, 12.0]),
             z2=np.zeros((2, 2)),
             z3=np.zeros((2, 2, 2)),
             f =0.5)
    ]

    # ------ sweep the drive frequency -----------------------------------
    ω_min, ω_max, n_ω = 2.0, 4.0, 300
    ω_grid = np.linspace(ω_min, ω_max, n_ω)

    current_time = time.time()
    
    amps = np.array([
        steady_state_amp(params, ω_d)
        for ω_d in ω_grid
    ])
    
    print(f"Elapsed time: {time.time() - current_time:.2f} seconds")

    # ------ visualize ----------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(ω_grid, amps, "-o", markersize=1)
    plt.xlabel(r"Drive frequency  $\omega_d$  [rad s$^{-1}$]")
    plt.ylabel(r"Steady-state amplitude  $|q_1|_{\max}$")
    plt.title("Frequency-response curve (SciPy integrator)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
