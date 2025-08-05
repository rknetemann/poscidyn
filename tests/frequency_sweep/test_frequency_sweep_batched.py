from jax import numpy as jnp
import numpy as np
import sys
import os
import jax
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

DUFFING_COEFFICIENTS = jnp.linspace(-0.005, 0.03, 3, dtype=oscidyn.const.DTYPE)  # Shape: (n_duffing,)
DRIVING_FREQUENCIES = jnp.linspace(0.1, 2.0, 500, dtype=oscidyn.const.DTYPE) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDES = jnp.linspace(0.01, 1.0, 10, dtype=oscidyn.const.DTYPE)  # Shape: (n_driving_amplitudes,)

@jax.jit
def batched_frequency_sweep(
    duffing: float,
):
    n_modes = 1
    omega_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
    x_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
    omega_0_hat = jnp.array([1.0], dtype=oscidyn.const.DTYPE)
    Q = jnp.array([10.0], dtype=oscidyn.const.DTYPE)
    eta_hat = jnp.array([0.005], dtype=oscidyn.const.DTYPE)
    alpha_hat = jnp.zeros((1, 1, 1), dtype=oscidyn.const.DTYPE)
    gamma_hat = jnp.zeros((1, 1, 1, 1), dtype=oscidyn.const.DTYPE).at[0, 0, 0, 0].set(duffing)
    delta_hat = jnp.zeros((1, 1, 1, 1, 1), dtype=oscidyn.const.DTYPE)

    model = oscidyn.NonlinearOscillator(
        n_modes=n_modes,
        Q=Q,
        eta_hat=eta_hat,
        alpha_hat=alpha_hat,
        gamma_hat=gamma_hat,
        delta_hat=delta_hat,
        omega_0_hat=omega_0_hat,
        omega_ref=omega_ref,
        x_ref=x_ref
    )
    
    return oscidyn.vmap_safe_frequency_sweep(
        model=model,
        sweep_direction=oscidyn.SweepDirection.FORWARD,
        driving_frequencies=DRIVING_FREQUENCIES,
        driving_amplitudes=DRIVING_AMPLITUDES,
        solver=oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, n_time_steps=1024, rtol=1e-4, atol=1e-6),
    )

frequency_sweeps = jax.vmap(batched_frequency_sweep)(DUFFING_COEFFICIENTS) # (n_duffing, n_freq * n_amp)

frequency_sweeps = frequency_sweeps.reshape(
    DUFFING_COEFFICIENTS.shape[0], 
    DRIVING_FREQUENCIES.shape[0], 
    DRIVING_AMPLITUDES.shape[0],
) # (n_duffing, n_freq, n_amp)

n_d = DUFFING_COEFFICIENTS.shape[0]
n_a = DRIVING_AMPLITUDES.shape[0]

# prepare grayscale colors (reversed)
gray_colors = plt.cm.gray(jnp.linspace(0.1, 0.9, n_a))[::-1]

# create one subplot per Duffing coefficient
fig, axes = plt.subplots(n_d, 1, sharex=True, figsize=(8, 4 * n_d))

# if there's only one subplot, wrap it in a list for consistency
if n_d == 1:
    axes = [axes]

for i, ax in enumerate(axes):
    for j, c in enumerate(gray_colors):
        ax.plot(
            DRIVING_FREQUENCIES,
            frequency_sweeps[i, :, j],
            color=c
        )
    ax.set_title(f"Duffing = {float(DUFFING_COEFFICIENTS[i]):.3g}")
    ax.set_xlabel("Driving frequency")
    ax.set_ylabel("Steady-state displacement amplitude")
    ax.grid(True)

plt.tight_layout()
plt.show()
plt.savefig("results/frequency_sweep_duffing.png", dpi=300, bbox_inches='tight')