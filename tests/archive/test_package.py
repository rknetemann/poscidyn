# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import emd
from scipy.signal import hilbert

import oscidyn
from scipy.signal import find_peaks

#from farbod_model import mdl_farbod


# ────────────── switches ────────────────────────────────────
RUN_TIME_RESPONSE = True
RUN_FREQUENCY_RESPONSE   = False
RUN_PHASE_SPACE = False


# ────────────── build & scale model ─────────────────────────
#mdl = oscidyn.NonDimensionalisedModel.from_example(1)
mdl = oscidyn.PhysicalModel.from_example(1)
print(mdl.omega_0)
mdl = mdl.non_dimensionalise()
#mdl = mdl_farbod
nld = oscidyn.NonlinearDynamics(mdl)

# =============== time-response ===================
if RUN_TIME_RESPONSE:
    print("\nCalculating time response...")
    
    # Define parameters for time response
    F_omega_hat_value = mdl.omega_0_hat[0] * 1  # Slightly below first natural frequency
    F_amp_hat_value = mdl.F_amp_hat               # Use model's default amplitude
    
    # Initial displacement (small perturbation) and zero velocity
    n_modes = mdl.N
    q0_hat = jnp.ones(n_modes) * 0.1              # Small initial displacement
    v0_hat = jnp.ones(n_modes) * 0                 # Zero initial velocity
    y0_hat = jnp.concatenate([q0_hat, v0_hat])    # Combined initial state
    
    # Calculate time response
    tau, q, v = nld.time_response(
        tau_end= 1000,
        F_omega_hat=jnp.array([F_omega_hat_value]),
        F_amp_hat=F_amp_hat_value,
        y0_hat=y0_hat,
        n_steps=4000,                             # More steps for smoother curves
        calculate_dimless=True                    # Use non-dimensional equations
    )
    
    q_fft = np.fft.fft2(q)
 
    q_fft_sort = np.sort(np.abs(q_fft), axis=0)
    keep = 0.3
    threshold = q_fft_sort[int(np.floor((1-keep) * q_fft_sort.shape[0])), :]
    ind = np.abs(q_fft) > threshold
    q_fft_compress = q_fft * ind
    q_compress = np.fft.ifft2(q_fft_compress).real

    # Plot original signal vs compressed signal
    fig_comp, axes_comp = plt.subplots(n_modes, 1, figsize=(10, 2.5 * n_modes), sharex=True)
    if n_modes == 1:
        axes_comp = [axes_comp]  # Handle the case of a single mode

    for i in range(n_modes):
        axes_comp[i].plot(tau, q[:, i], label='Original', color='#0072B2', linewidth=1.25)
        axes_comp[i].plot(tau, q_compress[:, i], label='Compressed', color='#D55E00', 
                          linewidth=1.25, linestyle='--')
        axes_comp[i].set_ylabel(f"$q_{{{i+1}}}$", fontsize=11)
        axes_comp[i].grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
        axes_comp[i].legend(loc='upper right', framealpha=0.7, edgecolor='none')
        axes_comp[i].spines['top'].set_visible(False)
        axes_comp[i].spines['right'].set_visible(False)

    axes_comp[-1].set_xlabel("Non-dimensional time $\\tau$", fontsize=11)
    plt.suptitle("Original vs Compressed Signal Comparison", fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust to make room for the suptitle
    plt.show()
    

    # Extract local maxima from the signal q

    # If q has multiple modes (columns), use the first mode for simplicity
    if q.ndim > 1:
        q_signal = q[:, 0]  # Use the first mode for analysis
    else:
        q_signal = q  # If q is already 1D

    # Convert from JAX array to NumPy array if needed
    if isinstance(q_signal, jnp.ndarray):
        q_signal = np.array(q_signal)
    if isinstance(tau, jnp.ndarray):
        tau = np.array(tau)

    # Find peaks (local maxima) in the signal
    peak_indices, _ = find_peaks(q_signal)
    peak_values = q_signal[peak_indices]
    peak_times = tau[peak_indices]

    # Calculate the envelope using Hilbert transform
    analytic_signal = hilbert(q_signal)
    envelope = np.abs(analytic_signal)

    # Print information about the peaks
    print(f"Found {len(peak_indices)} local maxima in the signal")
    print(f"Average peak value: {np.mean(peak_values):.4f}")
        
    # Create figure for plotting time response with professional style
    fig, axes = plt.subplots(n_modes + 1, 1, figsize=(7.5, 1.8 * (n_modes + 1)), sharex=True)
    
    # Use a publication-quality color palette
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
    
    # Plot each mode's displacement
    for i in range(n_modes):
        axes[i].plot(tau, q[:, i], label=f"Mode {i+1}", color=colors[i % len(colors)], linewidth=1.25)
        
        # Add peaks as dots for the first mode
        if i == 0:
            axes[i].scatter(peak_times, peak_values, color='red', s=30, marker='o', 
                          label='Peaks', zorder=5, alpha=0.7)
            
        axes[i].set_ylabel(f"$q_{{{i+1}}}$", fontsize=11)
        axes[i].grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
        axes[i].legend(loc='upper right', framealpha=0.7, edgecolor='none', 
                     handlelength=1.5, handletextpad=0.5)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.15, linewidth=0.8)
    
    # Plot total displacement
    q_total = jnp.sum(q, axis=1)
    axes[-1].plot(tau, q_total, label="Total", color='#000000', linewidth=1.5)
    
    # Add peaks to the total displacement plot (optional)
    total_peaks, _ = find_peaks(np.array(q_total))
    total_peak_times = tau[total_peaks]
    total_peak_values = q_total[total_peaks]
    axes[-1].scatter(total_peak_times, total_peak_values, color='red', s=30, marker='o', 
                   label='Peaks', zorder=5, alpha=0.7)
    
    axes[-1].set_ylabel("$q_{\\mathrm{total}}$", fontsize=11)
    axes[-1].set_xlabel("Non-dimensional time $\\tau$", fontsize=11)
    axes[-1].grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
    axes[-1].legend(loc='upper right', framealpha=0.7, edgecolor='none',
                  handlelength=1.5, handletextpad=0.5)
    axes[-1].spines['top'].set_visible(False)# Create figure for plotting time response
    
    # Optional: Phase space trajectory of first mode from time response data
    plt.figure(figsize=(8, 6))
    plt.plot(q[:, 0], v[:, 0], lw=0.8)
    plt.plot(q[0, 0], v[0, 0], 'go', label='Start')
    plt.plot(q[-1, 0], v[-1, 0], 'ro', label='End')
    plt.xlabel("Displacement $q_1$")
    plt.ylabel("Velocity $v_1$")
    plt.title(f"Phase Portrait of Mode 1 from Time Response at $\\hat{{\\omega}}_F = {F_omega_hat_value:.4f}$")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# =============== frequency sweep ===================
if RUN_FREQUENCY_RESPONSE:
    # F_omega_hat_grid = jnp.linspace(0.1, 4.0, 1000)  # Define a range of frequencies
    # tau_end = 300.0  # End time for the simulation
    # F_omega_hat_fw, q_steady_fw, q_steady_total_fw, _, phase_fw, _ = nld.frequency_response(sweep_direction=oscidyn.Sweep.FORWARD, F_omega_hat_grid=F_omega_hat_grid, tau_end=tau_end)
    # F_omega_hat_bw, q_steady_bw, q_steady_total_bw, _, phase_bw, _ = nld.frequency_response(sweep_direction=oscidyn.Sweep.BACKWARD, F_omega_hat_grid=F_omega_hat_grid, tau_end=tau_end)
    
    F_omega_hat_fw, q_steady_fw, q_steady_total_fw, _, phase_fw, _ = nld.frequency_response(sweep_direction=oscidyn.Sweep.FORWARD)
    F_omega_hat_bw, q_steady_bw, q_steady_total_bw, _, phase_bw, _ = nld.frequency_response(sweep_direction=oscidyn.Sweep.BACKWARD)

    
    # Create figure with 2 subplots - one for amplitude, one for phase
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors

    # Plot amplitude response on the first subplot
    for m in range(mdl.N):
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
    for m in range(mdl.N):
        # Forward sweep - convert phase to degrees for better readability
        ax2.plot(F_omega_hat_fw, np.rad2deg(phase_fw), label=f"Phase (Forward)", color="k", alpha=0.7)
        # Backward sweep
        ax2.plot(F_omega_hat_bw, np.rad2deg(phase_bw), label=f"Phase (Backward)", color="gray", alpha=0.4)

    ax2.set_xlabel("Non-dimensionalized drive frequency")
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_title("Frequency Response - Phase")
    ax2.set_ylim(-180, 180)
    ax2.grid(True)
    ax2.legend()

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