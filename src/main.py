import numpy as np
import time
import matplotlib.pyplot as plt

from modal_eom import ModalEOM

if __name__ == "__main__":
    # Example usage
    N = 2
    #modal_eom = ModalEOM.random(1, seed=33)
    modal_eom = ModalEOM.example()
    #modal_eom = ModalEOM.duffing()
    print(modal_eom.eigenfrequencies())
    
    omega_d_min, omega_d_max, n_omega_d = 3, 3.6, 500
    y0 = np.zeros(2*N)
    t_end = 500.0
    n_steps = 1000
    discard_frac = 0.9
    
    number_of_calculations = n_omega_d * n_steps
    print(f"Number of calculations: {number_of_calculations}")

    current_time = time.time()
    
    omega_d, amp = modal_eom.frequency_response(
        omega_d_min=omega_d_min,
        omega_d_max=omega_d_max,
        n_omega_d=n_omega_d,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        discard_frac=discard_frac
    )
    
    elapsed_time = time.time() - current_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds. ({(number_of_calculations / elapsed_time):.2f} calculations/s)")

    # ------ visualize ----------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(omega_d, amp, "-o", markersize=1)
    plt.xlabel(r"Drive frequency  $\omega_d$  [rad s$^{-1}$]")
    plt.ylabel(r"Steady-state amplitude  $|q_1|_{\max}$")
    plt.title("Frequency-response curve (SciPy integrator)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()