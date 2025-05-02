# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401

from model import Model

# ────────────── switches ────────────────────────────────────
RUN_TIME   = False     # single-tone time trace
RUN_FREQ   = True      # frequency-response curve
RUN_FORCE  = False     # force-sweep surface

# ────────────── build & scale model ─────────────────────────
N   = 1
mdl = Model.from_example(N)
#mdl = Model.from_random(N)
#print(mdl)

# ────────────── eigenfrequencies ─────────────────────────

eigenfreq = mdl.eigenfrequencies()

# =============== study 1: time response =====================
if RUN_TIME:
    print("\nCalculating time response …")
    y0      = jnp.zeros(2*N)
    t_end   = 50  
    f_omega_hz = 0.49

    ts, qs, _ = mdl.time_response(
        y0=y0, n_steps=4000,
        f_omega_hz=jnp.array([f_omega_hz]),
    )

    t_plot = ts[0]
    q_plot = qs[0]     

    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(t_plot, q_plot[:, m], label=f"Mode {m+1}")
    plt.xlabel("Dimensionless time");  plt.ylabel("Dimensionless amplitude")
    plt.title("Time response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# =============== study 2: frequency sweep ===================
if RUN_FREQ:
    print("\nCalculating frequency response …")
    f_omega_sweep_hz = jnp.linspace(0.0, 1.0, 400)
    
    f_omega_hz, q_steady, q_steady_total, _ = mdl.frequency_response(
        f_omega_hz=f_omega_sweep_hz, t_end=1000,
    )
    
    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(f_omega_hz, q_steady[:, m], label=f"Mode {m+1}")
    for f in eigenfreq:
        plt.axvline(f, ls="--", color="r", alpha=.6)
    plt.plot(f_omega_hz, q_steady_total, label="Total Response", color="k", lw=2, alpha=0.8)
    plt.xlabel("Non-dimensionalized drive frequency"); plt.ylabel("Non-dimensionalized amplitude")
    plt.title("Frequency response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# =============== study 3: force sweep =======================
if RUN_FORCE:
    print("\nCalculating force sweep …")
    f_amp = jnp.zeros(N).at[0].set(1.0) # only first mode
    f_amp_levels = jnp.array([0, 1, 5, 10, 15, 16.5, 16.9])

    f_omega_sweep_hz = jnp.linspace(0, 1, 400)
    y0_hat, t_end_hat = jnp.zeros(2*N), 50.0

    f_omega_dimless, amplitude_dimless = mdl.force_sweep(
        y0=y0_hat, t_end=t_end_hat, n_steps=4000,
        discard_frac=0.8, f_amp=f_amp, f_amp_levels=f_amp_levels, f_omega_hz=f_omega_sweep_hz
    )                                 

    # Create a single 3D plot showing all modes at each force level
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colormaps
    force_cmap = plt.cm.viridis(np.linspace(0, 1, len(f_amp_levels)))
    mode_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']  # Colors for each mode
    
    # Plot each force level as a plane with all modes
    for j, level in enumerate(f_amp_levels):
        for mode_idx in range(N):
            ax.plot(f_omega_dimless, 
                   np.zeros_like(f_omega_dimless) + j, 
                   amplitude_dimless[j, :, mode_idx],
                   color=mode_colors[mode_idx % len(mode_colors)], 
                   alpha=0.8,
                   linewidth=2)
            
    # Add a point at the beginning of each line to create the legend entries
    mode_handles = []
    for mode_idx in range(N):
        line, = ax.plot([f_omega_dimless[0]], [0], [0], 
                      color=mode_colors[mode_idx % len(mode_colors)],
                      linewidth=2,
                      label=f"Mode {mode_idx+1}")
        mode_handles.append(line)
    
    # Add force level annotations
    for j, level in enumerate(f_amp_levels):
        ax.text(f_omega_dimless[0], j, np.max(amplitude_dimless[j]), 
                f"F={level:.1f}", color=force_cmap[j], fontsize=10)
    
    ax.set_xlabel("Non-dimensionalized drive frequency")
    ax.set_ylabel("Force level index")
    ax.set_yticks(range(len(f_amp_levels)))
    ax.set_yticklabels([f"{level:.1f}" for level in f_amp_levels])
    ax.set_zlabel("Non-dimensionalized amplitude")
    ax.set_title("Force sweep – all modes")
    
    # Add legend for modes
    ax.legend(title="Modes")
    
    plt.tight_layout()
    plt.show()
# ------------------------------------------------------------------------
