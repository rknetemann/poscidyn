import matplotlib.pyplot as plt
import jax.numpy as jnp

from .. import constants as const

def plot_branch_exploration(coarse_drive_freq_mesh, coarse_drive_amp_mesh, y_max_disp):
    # flatten the coarse‐grid for plotting
    freq_vals = coarse_drive_freq_mesh.ravel()
    amp_vals  = coarse_drive_amp_mesh.ravel()
    disp_vals = jnp.abs(y_max_disp[..., 0]).ravel()

    fig, ax = plt.subplots(figsize=(8, 6))
    # background scatter in gray
    sc = ax.scatter(
        freq_vals,
        disp_vals,
        c=amp_vals,
        cmap='Greys',
        vmin=amp_vals.min(),
        vmax=amp_vals.max()
    )
    fig.colorbar(sc, ax=ax, label='Driving amplitude')
    ax.set_xlabel('Driving frequency')
    ax.set_ylabel('Steady-state displacement amplitude')
    ax.set_title('Branch Exploration')
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.show()

def plot_branch_selection(driving_frequencies, driving_amplitudes, ss_disp_amp):
    coarse_drive_freq = jnp.linspace(jnp.min(driving_frequencies), jnp.max(driving_frequencies), ss_disp_amp.shape[0]) # (n_coarse_freq,)
    coarse_drive_amp = jnp.linspace(jnp.min(driving_amplitudes), jnp.max(driving_amplitudes), ss_disp_amp.shape[1]) # (n_coarse_amp,)

    # number of coarse amplitudes
    n_a = coarse_drive_amp.shape[0]

    # generate a reversed gray‐scale palette from light to dark
    gray_colors = plt.cm.gray(jnp.linspace(0.1, 0.9, n_a))[::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    for ia, amp in enumerate(coarse_drive_amp):
        color = gray_colors[ia]
        # steady‐state displacement amplitude for mode 0 at this amplitude
        disp_curve = ss_disp_amp[:, ia, 0]
        ax.plot(coarse_drive_freq, disp_curve, color=color, label=f"A={amp:.2f}")

    ax.set_xlabel("Driving frequency")
    ax.set_ylabel("Steady-state displacement amplitude")
    ax.set_title("Branch Selection")
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.show()

def plot_interpolated_sweep(driving_frequencies, driving_amplitudes, disp_init):
    # plot initial displacement from y_init_fine for each amplitude
    n_amp = driving_amplitudes.shape[0]

    # y_init_fine has shape (n_freq, n_amp, n_state), first n_modes entries are displacements

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # generate grayscale colors and reverse the sequence - same as other functions
    gray_colors = plt.cm.gray(jnp.linspace(0.1, 0.9, n_amp))[::-1]
    
    for j, c in enumerate(gray_colors):
        # plot mode 0 displacement as an example; change index if you want other modes
        ax.plot(driving_frequencies, disp_init[:, j, 0], color=c, 
                label=f"amp={driving_amplitudes[j]:.2f}")
    ax.set_xlabel("Driving Frequency")
    ax.set_ylabel("Initial Displacement (mode 0)")
    ax.set_title("Interpolated Results")
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.show()

def plot_frequency_sweep(frequency_sweep):
    print(frequency_sweep)
    n_f = frequency_sweep.driving_frequencies.shape[0]
    n_a = frequency_sweep.driving_amplitudes.shape[0]
    amps = frequency_sweep.total_steady_state_displacement_amplitude.reshape(n_f, n_a)

    # 2D Line plots in reversed grayscale
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # generate grayscale colors and reverse the sequence
    gray_colors = plt.cm.gray(jnp.linspace(0.1, 0.9, n_a))[::-1]
    
    for j, c in enumerate(gray_colors):
        ax.plot(frequency_sweep.driving_frequencies, amps[:, j], color=c, 
                label=f"A={frequency_sweep.driving_amplitudes[j]:.2g}")
                
    ax.set_xlabel("Driving frequency")
    ax.set_ylabel("Total steady-state displacement amplitude")
    #ax.legend(title="Drive amplitude")  # Keeping this commented as in original
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.show()