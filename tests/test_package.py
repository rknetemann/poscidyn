# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import oscidyn

from farbod_model import mdl_farbod

# ────────────── switches ────────────────────────────────────
RUN_FREQUENCY_RESPONSE   = True
RUN_PHASE_SPACE = False

# ────────────── build & scale model ─────────────────────────
mdl = oscidyn.PhysicalModel.from_example(2).non_dimensionalise()
mdl = mdl_farbod
nld = oscidyn.NonlinearDynamics(mdl)
N = mdl.N

# =============== frequency sweep ===================
if RUN_FREQUENCY_RESPONSE:
    F_omega_hat_grid = jnp.linspace(0.1, 2.0, 1000)  # Define a range of frequencies
    F_omega_hat_fw, q_steady_fw, q_steady_total_fw, _, phase_fw, _ = nld.frequency_response(F_omega_hat_grid=F_omega_hat_grid, sweep_direction=oscidyn.Sweep.FORWARD)
    F_omega_hat_bw, q_steady_bw, q_steady_total_bw, _, phase_bw, _ = nld.frequency_response(F_omega_hat_grid=F_omega_hat_grid, sweep_direction=oscidyn.Sweep.BACKWARD)
    
    # Create figure with 2 subplots - one for amplitude, one for phase
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    # Plot amplitude response on the first subplot
    for m in range(N):
        # Forward sweep
        ax1.plot(F_omega_hat_fw, q_steady_fw[:, m], label=f"Mode {m+1} (Forward)", color=colors[m % len(colors)], alpha=0.7)
        # Backward sweep
        ax1.plot(F_omega_hat_bw, q_steady_bw[:, m], label=f"Mode {m+1} (Backward)", color=colors[m % len(colors)], alpha=0.4)

    for f in mdl.omega_0_hat:
        ax1.axvline(f, ls="--", color="r", alpha=0.6)
        ax2.axvline(f, ls="--", color="r", alpha=0.6)  # Add to phase plot too

    # Total response
    ax1.plot(F_omega_hat_fw, q_steady_total_fw, label="Total Response (Forward)", color="k", lw=2, alpha=0.8)
    ax1.plot(F_omega_hat_bw, q_steady_total_bw, label="Total Response (Backward)", color="gray", lw=2, alpha=0.5)

    ax1.set_ylabel("Non-dimensionalized amplitude")
    ax1.set_title("Frequency Response - Amplitude")
    ax1.grid(True)
    ax1.legend()

    # Plot phase response on the second subplot
    for m in range(N):
        # Forward sweep - convert phase to degrees for better readability
        ax2.plot(F_omega_hat_fw, np.rad2deg(phase_fw), label=f"Phase (Forward)", color="k", alpha=0.7)
        # Backward sweep
        ax2.plot(F_omega_hat_bw, np.rad2deg(phase_bw), label=f"Phase (Backward)", color="gray", alpha=0.4)

    ax2.set_xlabel("Non-dimensionalized drive frequency")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Frequency Response - Phase")
    ax2.set_ylim(-180, 180)
    ax2.grid(True)
    #ax2.legend()

    plt.tight_layout()
    plt.show()

# =============== phase space plot ===================
if RUN_PHASE_SPACE:
    print("\nCalculating phase portrait...")
    
    # Select a forcing frequency (e.g., near a resonance or from F_oega_hat array)
    # For demonstration, let's pick the model's default F_omega_hat if available,
    # or a value known to be interesting (e.g., first eigenfrequency).
    if mdl.F_omega_hat is not None and mdl.F_omega_hat.size > 0:
        F_omega_hat_single = mdl.F_omega_hat[0] 
    else:
        F_omega_hat_single = mdl.omega_0_hat[0] # Example: use first eigenfrequency

    # You can also pick a frequency from the `F_omega_hat` array from the frequency response
    # For example, if RUN_FREQ was true:
    # F_omega_hat_single = F_omega_hat[len(F_omega_hat) // 2] # Middle frequency
    
    print(f"-> Using F_omega_hat = {F_omega_hat_single:.4f} for phase portrait.")

    tau_phase, q_phase, v_phase = nld.phase_portrait(
        F_omega_hat=jnp.array([F_omega_hat_single]), # Ensure it's an array
        # F_amp_hat can be default from model, or specified:
        # F_amp_hat=mdl.F_amp_hat 
        y0_hat=jnp.zeros(2 * N), # Start from rest
        tau_end=500.0,           # Simulate for a longer time to see attractor
        n_steps=8000
    )
    
    # Plot phase portraits for each mode
    # Determine the number of rows and columns for subplots
    n_modes = q_phase.shape[1]
    if n_modes == 1:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        axs = [ax]
    else:
        n_cols = 2
        n_rows = (n_modes + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        axs = axs.flatten()

    for i in range(n_modes):
        axs[i].plot(q_phase[:, i], v_phase[:, i], lw=0.5)
        axs[i].set_xlabel(f"Displacement $q_{i+1}$ (non-dim.)")
        axs[i].set_ylabel(f"Velocity $v_{i+1}$ (non-dim.)")
        axs[i].set_title(f"Phase Portrait - Mode {i+1}")
        axs[i].grid(True)
        
        # Mark the start and end points
        axs[i].plot(q_phase[0, i], v_phase[0, i], 'go', label='Start') # Start point
        axs[i].plot(q_phase[-1, i], v_phase[-1, i], 'ro', label='End')   # End point
        if i == 0: # Add legend to the first plot
            axs[i].legend()

    # Hide any unused subplots
    for j in range(n_modes, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(f"Phase Portraits at $\\hat{{\\omega}}_F = {F_omega_hat_single:.4f}$", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()