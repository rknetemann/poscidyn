from jax import numpy as jnp
import numpy as np
import sys
import os
import jax
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

N_DUFFING = 40
DUFFING_COEFFICIENTS = jnp.linspace(-0.005, 0.03, N_DUFFING, dtype=oscidyn.const.DTYPE)  # Shape: (n_duffing,)
DRIVING_FREQUENCIES = jnp.linspace(0.1, 2.0, 500, dtype=oscidyn.const.DTYPE) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDES = jnp.linspace(0.01, 1.0, 10, dtype=oscidyn.const.DTYPE)  # Shape: (n_driving_amplitudes,)

start_time = time.time()

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


# split Duffing coefficients into 4 parts
n_d = DUFFING_COEFFICIENTS.shape[0]
quarter = n_d // 4
duff_q1 = DUFFING_COEFFICIENTS[:quarter]
duff_q2 = DUFFING_COEFFICIENTS[quarter:2*quarter]
duff_q3 = DUFFING_COEFFICIENTS[2*quarter:3*quarter]
duff_q4 = DUFFING_COEFFICIENTS[3*quarter:]

# run batched sweep on each quarter
sweeps_q1 = jax.vmap(batched_frequency_sweep)(duff_q1)
sweeps_q2 = jax.vmap(batched_frequency_sweep)(duff_q2)
sweeps_q3 = jax.vmap(batched_frequency_sweep)(duff_q3)
sweeps_q4 = jax.vmap(batched_frequency_sweep)(duff_q4)

# combine results
frequency_sweeps = jnp.concatenate([sweeps_q1, sweeps_q2, sweeps_q3, sweeps_q4], axis=0)

print(f"Time taken: {time.time() - start_time:.2f} seconds")
end_time = time.time()
elapsed = end_time - start_time
simulations_per_second = N_DUFFING / elapsed
print(f"Simulations per second: {simulations_per_second:.2f}")

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