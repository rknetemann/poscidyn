# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import oscidyn

mdl = oscidyn.PhysicalModel.from_example(1).non_dimensionalise()
nld = oscidyn.NonlinearDynamics(mdl)


print("\nCalculating time response...")

# Define parameters for time response
F_omega_hat_value = mdl.omega_0_hat[0] * 1  # Slightly below first natural frequency
F_amp_hat_value = mdl.F_amp_hat*2               # Use model's default amplitude

# Initial displacement (small perturbation) and zero velocity
n_modes = mdl.N
q0_hat = jnp.ones(n_modes) * 0.0              # Small initial displacement
v0_hat = jnp.ones(n_modes) * 0.0                 # Zero initial velocity
y0_hat = jnp.concatenate([q0_hat, v0_hat])    # Combined initial state

# Calculate time response
tau, q, v = nld.time_response(
    F_omega_hat=jnp.array([F_omega_hat_value]),
    F_amp_hat=F_amp_hat_value,
    y0_hat=y0_hat,
    n_steps=4000,                             # More steps for smoother curves
    calculate_dimless=True                    # Use non-dimensional equations
)

# Create figure for plotting time response with professional style
fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))

# Use a publication-quality color palette
colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']

# Plot total displacement only
q_total = jnp.sum(q, axis=1)
ax.plot(tau, q_total, label="Total Displacement", color='#000000', linewidth=1.0)
ax.set_ylabel("Displacement x", fontsize=11)
ax.set_xlabel("Time t", fontsize=11)
ax.grid(True, linestyle=':', alpha=0.6, linewidth=0.5)
#ax.legend(loc='upper right', framealpha=0.7, edgecolor='none',
#          handlelength=1.5, handletextpad=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.15, linewidth=0.8)

# Add forcing frequency information to the title
#fig.suptitle(f"Time Response at $\\hat{{\\omega}}_F = {F_omega_hat_value:.2f}$", fontsize=13)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
plt.show()