import numpy as np
import jax.numpy as jnp
import time
import matplotlib.pyplot as plt

from modal_eom import ModalEOM
from modal_eom_diffrax import ModalEOM as ModalEOMDiffrax
from modal_eom_improved import ModalEOM as ModalEOMImproved

if __name__ == "__main__":
    N = 2
    modal_eom = ModalEOMImproved.from_example()
    #modal_eom = ModalEOMImproved.from_random(N, seed=0)
    
    eig_freqs = modal_eom.eigenfrequencies()
    print("Eigenfrequencies (Hz):")
    for idx, freq in enumerate(eig_freqs, start=1):
        print(f"- Mode {idx}: {(freq / (2 * np.pi)):.2f} Hz")
    
    f_omega_min, f_omega_max, f_omega_n = 0, 1 * 2 * jnp.pi, 1000
    f_omega = jnp.linspace(f_omega_min, f_omega_max, f_omega_n)
    
    f_amp_min, f_amp_max, f_amp_n = 0, 15, 15
    f_amp = jnp.linspace(f_amp_min, f_amp_max, f_amp_n) 
    
    y0 = jnp.zeros(2*N)
    t_end = 100.0
    n_steps = 500
    discard_frac = 0.8
    
    number_of_calculations = f_omega_n * n_steps
    print(f"\nNumber of calculations: {number_of_calculations}")

    # print("\nCalculating time response...")
    # current_time = time.time()
    # time_response = modal_eom.time_response(
    #     y0=y0,
    #     t_end=t_end,
    #     n_steps=n_steps,
    # )
    # elapsed_time = time.time() - current_time
    # print(f"-> Time response elapsed time: {elapsed_time:.2f} seconds. ({(number_of_calculations / elapsed_time):.2f} calculations/s)")
    
    print("\nCalculating frequency response...")
    current_time = time.time()
    omega_d, frequency_response = modal_eom.frequency_response(
        f_omega=f_omega,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        discard_frac=discard_frac
    )
    elapsed_time = time.time() - current_time
    print(f"-> Frequency response elapsed time: {elapsed_time:.2f} seconds. ({(number_of_calculations / elapsed_time):.2f} calculations/s)")
    
    # ------ visualize ----------------------------------------------------

    # # plot time response
    # plt.figure(figsize=(7, 4))
    # plt.plot(time_response[0], label="Time Response")
    # plt.xlabel("Time")
    # plt.ylabel("Amplitude")
    # plt.title("Time Response of the System")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # plot frequency response
    plt.figure(figsize=(7, 4))
    plt.plot(omega_d, frequency_response, "-o", markersize=1)
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
    
    