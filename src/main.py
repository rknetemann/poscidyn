# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401

from models import PhysicalModel, NonDimensionalisedModel
from nonlinear_dynamics import NonlinearDynamics

# ────────────── switches ────────────────────────────────────
RUN_TIME   = False     # single-tone time trace
RUN_FREQ   = True      # frequency-response curve
RUN_FORCE  = False     # force-sweep surface
RUN_PHASE_SPACE = False # phase space plot

# ────────────── build & scale model ─────────────────────────
N   = 1
mdl = PhysicalModel.from_example(N).non_dimensionalise()
#mdl = PhysicalModel.from_random(N).non_dimensionalise()
nld = NonlinearDynamics(mdl)
#mdl = Model.from_random(N)

# ────────────── eigenfrequencies ─────────────────────────

eigenfreq = mdl.omega_0_hat
quality_factors = mdl.Q
x_ref = mdl.x_ref
print(f"Eigenfrequencies: {eigenfreq}")
print(f"Quality factors: {quality_factors}")
print(f"Non-dimensionalised x_ref: {x_ref}")

# =============== frequency sweep ===================
if RUN_FREQ:
    print("\nCalculating frequency response…")
    F_omega_hat, q_steady, q_steady_total, _ = nld.frequency_response()
    
    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(F_omega_hat, q_steady[:, m], label=f"Mode {m+1}")
    for f in eigenfreq:
        plt.axvline(f, ls="--", color="r", alpha=.6)
    plt.plot(F_omega_hat, q_steady_total, label="Total Response", color="k", lw=2, alpha=0.8)
    plt.xlabel("Non-dimensionalized drive frequency"); plt.ylabel("Non-dimensionalized amplitude")
    plt.title("Frequency response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# =============== phase space plot ===================
if RUN_PHASE_SPACE:
    print("\nCalculating phase portrait...")
    
    # Select a forcing frequency (e.g., near a resonance or from F_omega_hat array)
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