import numpy as np
import time
import matplotlib.pyplot as plt

from modal_eom import ModalEOM
from modal_eom_diffrax import ModalEOM as ModalEOMDiffrax

if __name__ == "__main__":
    # Example usage
    N = 2
    #modal_eom = ModalEOM.random(N, seed=33)
    #modal_eom = ModalEOM.example()
    #modal_eom = ModalEOM.duffing()
    
    modal_eom = ModalEOMDiffrax.example()
    #modal_eom = ModalEOMDiffrax.random(N, seed=33)
    #modal_eom = ModalEOMDiffrax.duffing()
    
    eig_freqs = modal_eom.eigenfrequencies()
    print("Eigenfrequencies (Hz):")
    for idx, freq in enumerate(eig_freqs, start=1):
        print(f"- Mode {idx}: {(freq / (2 * np.pi)):.2f} Hz")
    
    driving_freq_min, driving_freq_max, driving_freq_n = 0, 1, 1000
    y0 = np.zeros(2*N)
    t_end = 250.0
    n_steps = 5000
    discard_frac = 0.9
    
    number_of_calculations = driving_freq_n * n_steps
    print(f"\nNumber of calculations: {number_of_calculations}")

    current_time = time.time()
    
    omega_d, amp = modal_eom.frequency_response(
        omega_d_min=driving_freq_min*2*np.pi,
        omega_d_max=driving_freq_max*2*np.pi,
        n_omega_d=driving_freq_n,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        discard_frac=discard_frac
    )
    
    elapsed_time = time.time() - current_time
    print(f"\nElapsed time: {elapsed_time:.2f} seconds. ({(number_of_calculations / elapsed_time):.2f} calculations/s)")

    # ------ visualize ----------------------------------------------------
    plt.figure(figsize=(7, 4))
    plt.plot(omega_d, amp, "-o", markersize=1)
    # add eigenfrequency lines
    for idx, f in enumerate(eig_freqs):
        plt.axvline(
            f,
            color='r',
            linestyle='--',
            alpha=0.7,
            label='Eigenfrequency' if idx == 0 else ""
        )
    plt.legend()
    plt.xlabel(r"Drive frequency  $\omega_d$  [rad s$^{-1}$]")
    plt.ylabel(r"Steady-state amplitude  $|q_1|_{\max}$")
    plt.title("Frequency-response curve (SciPy integrator)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()