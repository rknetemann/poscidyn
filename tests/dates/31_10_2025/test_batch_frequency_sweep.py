import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt

import oscidyn

N_DUFFING = 20
N_DUFFING_IN_PARALLEL = 2

Q, omega_0, alpha, gamma = np.array([50.0, 23.0, 23.0]), np.array([1.0, 2.0, 3.0]), np.zeros((3,3,3)), np.zeros((3,3,3,3))
gamma[1,1,1,1] = 0.3
gamma[2,2,2,2] = 0.3
gamma[0,0,1,1] = 0.1
gamma[1,0,0,1] = 0.1

DUFFING_COEFFICIENTS = np.array([0.03, 0.04, 0.05, 0.06])
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(0.01, 3.9, 351)
DRIVING_AMPLITUDE = np.linspace(0.1, 20.0, 5) * omega_0[0]**2/Q[0]
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(7, 7), linear_response_factor=20.2)
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=200, max_steps=4096*20, multistart=MULTISTART, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

start_time = time.time()

@jax.jit
def batched_frequency_sweep(
    duffing: jax.Array,
):
    Q, omega_0, alpha, gamma = jnp.array([50.0, 23.0, 23.0]), jnp.array([1.0, 2.0, 3.0]), jnp.zeros((3,3,3)), jnp.zeros((3,3,3,3))
    gamma = gamma.at[0, 0, 0, 0].set(duffing)
    model = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
    
    return oscidyn.frequency_sweep(
        model = model,
        sweep_direction = SWEEP_DIRECTION,
        driving_frequencies = DRIVING_FREQUENCY,
        driving_amplitudes = DRIVING_AMPLITUDE,
        solver = SOLVER,
        precision = PRECISION,
    )

# Process N_DUFFING_IN_PARALLEL coefficients at a time
n_d = DUFFING_COEFFICIENTS.shape[0]
n_batches = (n_d + N_DUFFING_IN_PARALLEL - 1) // N_DUFFING_IN_PARALLEL  # Ceiling division

for i in range(n_batches):
    start_idx = i * N_DUFFING_IN_PARALLEL
    end_idx = min(start_idx + N_DUFFING_IN_PARALLEL, n_d)
    
    batch_duffing = DUFFING_COEFFICIENTS[start_idx:end_idx]
    batch_sweeps = jax.vmap(batched_frequency_sweep)(batch_duffing)

print(f"Time taken: {time.time() - start_time:.2f} seconds")
end_time = time.time()
elapsed = end_time - start_time
simulations_per_second = N_DUFFING / elapsed
print(f"Simulations per second: {simulations_per_second:.2f}")
