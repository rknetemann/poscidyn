import jax
import jax.numpy as jnp
import math
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

driving_frequencies = jnp.linspace(0.1, 2.0, 200)
driving_amplitudes = jnp.linspace(0.01, 1.0, 10)

Q = jnp.linspace(1.1, 100.0, 100)  
gamma = jnp.linspace(0.01, 1.0, 100)  
sweep_direction = jnp.array([-1, 1])

n_modes = 1
omega_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
x_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
omega_0 = jnp.array([1.0],   dtype=oscidyn.const.DTYPE)
eta = jnp.array([0.005], dtype=oscidyn.const.DTYPE)
alpha0 = jnp.zeros((1, 1, 1), dtype=oscidyn.const.DTYPE)
delta0 = jnp.zeros((1, 1, 1, 1, 1), dtype=oscidyn.const.DTYPE)

Q, gamma, sweep_direction = jnp.meshgrid(Q, gamma, sweep_direction)
Q = Q.flatten()
gamma = gamma.flatten()
sweep_direction = sweep_direction.flatten()

n_sim = len(Q)
n_parallel_sim_per_device = 10
n_devices = jax.device_count()
n_sub_batches = math.ceil(n_sim / (n_parallel_sim_per_device * n_devices)) # Example: 1001 simulations, 10 simulations per device, 4 devices -> 26 sub-batches

def split_sub_batches(Q, gamma, sweep_direction, n_devices):
    params = jnp.column_stack((Q, gamma, sweep_direction))
    params_shards = jnp.array_split(params, n_devices)
    return jnp.array(params_shards)

def simulate(params): # params: (n_params,)
    Q_val, gamma_val, direction = params
    
    Q       = jnp.array([Q_val], dtype=oscidyn.const.DTYPE)
    gamma   = jnp.zeros((1, 1, 1, 1), dtype=oscidyn.const.DTYPE)
    gamma   = gamma.at[0, 0, 0, 0].set(gamma_val)

    model = oscidyn.NonlinearOscillator(
        n_modes=n_modes,
        Q=Q,
        eta_hat=eta,
        alpha_hat=alpha0,
        gamma_hat=gamma,
        delta_hat=delta0,
        omega_0_hat=omega_0,
        omega_ref=omega_ref,
        x_ref=x_ref
    )
    
    solver = oscidyn.FixedTimeSteadyStateSolver(
        max_steps=4096, n_time_steps=512, rtol=1e-4, atol=1e-6, progress_bar=False
    )

    return oscidyn.vmap_safe_frequency_sweep(
        model=model,
        sweep_direction=direction,
        driving_frequencies=driving_frequencies,
        driving_amplitudes=driving_amplitudes,
        solver=solver,
    )
    
params_shards = split_sub_batches(Q, gamma, sweep_direction, n_devices) # (n_devices, n_sim/n_devices, n_params)

simulate_sub_batch = jax.vmap(simulate) # input args: (n_sims_per_device, n_params)
parallel_sub_batch_simulate = jax.pmap(simulate_sub_batch, axis_name="devices") # input args: (n_devices, n_sims_per_device, n_params)


# Make sure we have a plain Python list of shard arrays
params_shards_list = [jnp.asarray(s) for s in split_sub_batches(Q, gamma, sweep_direction, n_devices)]

for i in tqdm(range(n_sub_batches)):
    device_batches = []
    start = i * n_parallel_sim_per_device
    end = start + n_parallel_sim_per_device

    for shard in params_shards_list:
        shard_len = shard.shape[0]

        if start >= shard_len:
            # Nothing left in this shard; pad with the last available row (or zeros if empty)
            if shard_len > 0:
                batch = jnp.repeat(shard[-1:, :], n_parallel_sim_per_device, axis=0)
            else:
                batch = jnp.zeros((n_parallel_sim_per_device, shard.shape[1]), dtype=oscidyn.const.DTYPE)
        else:
            # Take the slice and pad to fixed size if needed
            chunk = shard[start:min(end, shard_len)]
            if chunk.shape[0] < n_parallel_sim_per_device:
                pad_count = n_parallel_sim_per_device - chunk.shape[0]
                pad_row = chunk[-1:, :]  # repeat the last row
                chunk = jnp.vstack([chunk, jnp.repeat(pad_row, pad_count, axis=0)])
            batch = chunk

        device_batches.append(batch)

    sub_batch_params_shards = jnp.stack(device_batches, axis=0)  # (n_devices, n_parallel_sim_per_device, n_params)
    result = parallel_sub_batch_simulate(sub_batch_params_shards)
    # TODO: collect/trim `result` if you want to discard padded repeats.
