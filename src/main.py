import numpy as np
import time
import matplotlib.pyplot as plt

from normal_form import NormalForm
from solver import steady_state_amp

if __name__ == "__main__":
    # Example usage
    N = 1
    nf = NormalForm.random(N, seed=33)
    #nf = NormalForm.example()
    y0 = np.zeros(2*N)
    t_end = 2000.0
    n_steps = 4000
    discard_frac = 0.9
    
    omega_min, omega_max, n_omega = 0.0, 2000000.0, 10000
    omega_grid = np.linspace(omega_min, omega_max, n_omega)

    number_of_calculations = n_omega * n_steps
    print(f"Number of calculations: {number_of_calculations}")

    current_time = time.time()
    
    amps = np.array([
        steady_state_amp(nf, omega_d, y0, t_end, n_steps, discard_frac)
        for omega_d in omega_grid
    ])
    
    elapsed_time = time.time() - current_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds. ({(number_of_calculations / elapsed_time):.2f} calculations/s)")

    # ------ visualize ----------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(omega_grid, amps, "-o", markersize=1)
    plt.xlabel(r"Drive frequency  $\omega_d$  [rad s$^{-1}$]")
    plt.ylabel(r"Steady-state amplitude  $|q_1|_{\max}$")
    plt.title("Frequency-response curve (SciPy integrator)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()