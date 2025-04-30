import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", False)
import time
import matplotlib.pyplot as plt

from modal_eom import ModalEOM
from modal_eom_diffrax import ModalEOM as ModalEOMDiffrax
from modal_eom_improved import ModalEOM as ModalEOMImproved
from mpl_toolkits.mplot3d import Axes3D      # added for 3D plotting

if __name__ == "__main__":
    N = 2
    modal_eom = ModalEOMImproved.from_example(N)
    #modal_eom = ModalEOMImproved.from_random(N, seed=33)
    
    eig_freqs = modal_eom.eigenfrequencies()
    print("Eigenfrequencies (Hz):")
    for idx, freq in enumerate(eig_freqs, start=1):
        print(f"- Mode {idx}: {(freq / (2 * np.pi)):.2f} Hz")
    
    f_omega_min, f_omega_max, f_omega_n = 0 * 2 * jnp.pi, 1 * 2 * jnp.pi, 1000
    f_omega_sweep = jnp.linspace(f_omega_min, f_omega_max, f_omega_n)
    f_amp_sweep = jnp.array([[0.0, 0.5],[1.0, 0.5], [5.0, 0.5], [10.0, 0.5], [15.0, 0.5], [16.5, 0.5], [16.9, 0.5]])
    f_omega_time_response = jnp.array([0.49 * 2 * np.pi])
    
    y0 = jnp.zeros(2*N)
    t_end = 100.0
    n_steps = 500
    discard_frac = 0.8
    
    number_of_calculations = f_omega_n * n_steps
    print(f"\nNumber of calculations: {number_of_calculations}")

    print("\nCalculating time response...")
    ts, qs, vs = modal_eom.time_response(
        f_omega=f_omega_time_response,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
    )
    
    print("\nCalculating frequency response...")
    omega_d, q_steady, v_steady = modal_eom.frequency_response(
        f_omega=f_omega_sweep,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        discard_frac=discard_frac
    )
    
    print("\nCalculating force sweep...")
    omega_d, q_steady_forces= modal_eom.force_sweep(
        f_omega=f_omega_sweep,
        f_amp=f_amp_sweep,
        y0=y0,
        t_end=t_end,
        n_steps=n_steps,
        discard_frac=discard_frac
    )
    # swap axes to have shape (n_omega, n_forces, n_modes)
    frequency_response_force = q_steady_forces.transpose(1, 0, 2)
        
    # ------ visualize ----------------------------------------------------

    # plot time response
    plt.figure(figsize=(7, 4))                 # unpack times and modal amplitudes
    for idx in range(qs.shape[0]):
        plt.plot(ts[0], qs[idx], label=f"Mode {idx+1}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Time Response of the System")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot frequency response
    plt.figure(figsize=(7, 4))
    omega_d_hz = omega_d / (2 * np.pi)
    eig_freqs_hz = eig_freqs / (2 * np.pi)
    for idx in range(q_steady.shape[1]):
        plt.plot(omega_d_hz, q_steady[:, idx], "-o", markersize=1, label=f"Mode {idx+1}")
    for idx, f in enumerate(eig_freqs_hz):
        plt.axvline(
            f,
            color='r',
            linestyle='--',
            alpha=0.7,
            label='Eigenfrequency' if idx == 0 else ""
        )
    plt.legend()
    plt.xlabel(r"Drive frequency  $f_d$  [Hz]")
    plt.ylabel(r"Steady-state amplitude  $|q_1|_{\max}$")
    plt.title("Frequency-response curve")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # plot force sweep with separate mode‐planes and force lines
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection='3d')

    # prepare data
    omega_d_hz = omega_d / (2 * np.pi)
    f_amps = np.array(f_amp_sweep)[:, 0]
    n_modes = frequency_response_force.shape[2]
    n_forces = len(f_amps)
    # choose a colormap for the forces
    cmap = plt.cm.viridis(np.linspace(0, 1, n_forces))

    # for each mode, draw a constant‐y plane (mode index)
    for mode_idx in range(n_modes):
        y_plane = np.full_like(omega_d_hz, mode_idx, dtype=float)
        for i, (force, color) in enumerate(zip(f_amps, cmap)):
            zs = frequency_response_force[:, i, mode_idx]
            ax.plot(
                omega_d_hz,
                y_plane,
                zs,
                color=color,
                label=f"{force:.2f}" if mode_idx == 0 else ""
            )

    # axis labels and ticks
    ax.set_xlabel("Drive frequency $f_d$ [Hz]")
    ax.set_ylabel("Mode index")
    ax.set_zlabel("Steady‐state amplitude")
    ax.set_yticks(np.arange(n_modes))
    ax.set_yticklabels([f"Mode {i+1}" for i in range(n_modes)])

    # legend for force amplitudes
    ax.legend(title="Force amp", loc="upper left", bbox_to_anchor=(1.05, 1))

    ax.set_title("Force‐sweep frequency‐response lines per mode")
    plt.tight_layout()
    plt.show()

    # separate 2D plot for the first mode across all force amplitudes
    plt.figure(figsize=(7, 4))
    for i, force in enumerate(f_amps):
        plt.plot(
            omega_d_hz,
            frequency_response_force[:, i, 0],
            color=cmap[i],
            label=f"Force {force:.2f}"
        )
    plt.xlabel("Drive frequency $f_d$ [Hz]")
    plt.ylabel("Steady‐state amplitude (Mode 1)")
    plt.title("Force‐sweep response for Mode 1")
    plt.legend(title="Force amp")
    plt.grid(True)
    plt.tight_layout()
    plt.show()