from jax import numpy as jnp
import numpy as np
import sys
import os
import jax
import time
import matplotlib.pyplot as plt
import tensorstore as ts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

N_DUFFING = 20
N_DUFFING_IN_PARALLEL = 10

DUFFING_COEFFICIENTS = jnp.linspace(-0.005, 0.03, N_DUFFING, dtype=oscidyn.const.DTYPE)  # Shape: (n_duffing,)
DRIVING_FREQUENCIES = jnp.linspace(0.1, 2.0, 200, dtype=oscidyn.const.DTYPE) # Shape: (n_driving_frequencies,)
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
        solver=oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, n_time_steps=512, rtol=1e-4, atol=1e-6),
    )

# Process N_DUFFING_IN_PARALLEL coefficients at a time
n_d = DUFFING_COEFFICIENTS.shape[0]
n_batches = (n_d + N_DUFFING_IN_PARALLEL - 1) // N_DUFFING_IN_PARALLEL  # Ceiling division

for i in range(n_batches):
    start_idx = i * N_DUFFING_IN_PARALLEL
    end_idx = min(start_idx + N_DUFFING_IN_PARALLEL, n_d)
    
    batch_duffing = DUFFING_COEFFICIENTS[start_idx:end_idx]
    print(batch_duffing.shape)
    batch_sweeps = jax.vmap(batched_frequency_sweep)(batch_duffing)

print(f"Time taken: {time.time() - start_time:.2f} seconds")
end_time = time.time()
elapsed = end_time - start_time
simulations_per_second = N_DUFFING / elapsed
print(f"Simulations per second: {simulations_per_second:.2f}")
