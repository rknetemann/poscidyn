import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from modal_eom import ModalEOM
from modal_eom_diffrax import ModalEOM as ModalEOMDiffrax
from modal_eom_improved import ModalEOM as ModalEOMImproved

if __name__ == "__main__":
    # Example usage
    N = 2
    #modal_eom = ModalEOM.from_random(N, seed=33)
    #modal_eom = ModalEOM.from_example()
    #modal_eom = ModalEOM.from_duffing()
    
    modal_eom = ModalEOMImproved.from_example()
    #modal_eom = ModalEOMDiffrax.from_random(N, seed=33)
    #modal_eom = ModalEOMDiffrax.from_duffing()
    
    eig_freqs = modal_eom.eigenfrequencies()
    print("Eigenfrequencies (Hz):")
    for idx, freq in enumerate(eig_freqs, start=1):
        print(f"- Mode {idx}: {(freq / (2 * np.pi)):.2f} Hz")
    
    f_omega_min, f_omega_max, f_omega_n = 0, 1, 1000
    y0 = jnp.zeros(2*N)
    t_end = 500.0
    n_steps = 3000
    discard_frac = 0.9
    
    number_of_calculations = driving_freq_n * n_steps
    print(f"\nNumber of calculations: {number_of_calculations}")

    current_time = time.time()
    
    omega_d, amp = modal_eom.frequency_response(
        f_omega=jnp.linspace(f_omega_min, f_omega_max, f_omega_n),
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
    
    