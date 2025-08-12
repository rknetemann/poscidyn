import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Disable preallocation to avoid OOM errors

import jax
import jax.numpy as jnp
import math

import sys
from tqdm import tqdm
import subprocess
import time

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
n_parallel_sim_per_device = 15
n_devices = jax.device_count()
n_sub_batches = math.ceil(n_sim / (n_parallel_sim_per_device * n_devices)) # Example: 1001 simulations, 10 simulations per device, 2 devices -> 51 sub-batches

print(f"Total simulations: {n_sim}, Devices: {n_devices}, Parallel simulations per device: {n_parallel_sim_per_device}, Sub-batches: {n_sub_batches}")

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

def _gpu_postfix_smi():
    try:
        # Query GPU id, util %, memory used, memory total (CSV, no header)
        smi_out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ], encoding="utf-8")
        lines = smi_out.strip().split("\n")
        parts = []
        for line in lines:
            gid, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
            parts.append(f"GPU{gid} {util:>3}% {int(mem_used)/1024:.1f}GB/{int(mem_total)/1024:.1f}GB")
        return " | ".join(parts)
    except Exception:
        return ""
    
params_shards = split_sub_batches(Q, gamma, sweep_direction, n_devices) # (n_devices, n_sim/n_devices, n_params)

simulate_sub_batch = jax.vmap(simulate) # input args: (n_sims_per_device, n_params)
parallel_sub_batch_simulate = jax.pmap(simulate_sub_batch, axis_name="devices") # input args: (n_devices, n_sims_per_device, n_params)


params_shards_list = [jnp.asarray(s) for s in split_sub_batches(Q, gamma, sweep_direction, n_devices)]

last_postfix_time = 0.0
with tqdm(range(n_sub_batches), desc="Simulating", unit="sub") as pbar:
    for i in pbar:
        device_batches = []
        start = i * n_parallel_sim_per_device
        end = start + n_parallel_sim_per_device

        for shard in params_shards_list:
            shard_len = shard.shape[0]

            if start >= shard_len:
                if shard_len > 0:
                    batch = jnp.repeat(shard[-1:, :], n_parallel_sim_per_device, axis=0)
                else:
                    batch = jnp.zeros((n_parallel_sim_per_device, shard.shape[1]), dtype=oscidyn.const.DTYPE)
            else:
                chunk = shard[start:min(end, shard_len)]
                if chunk.shape[0] < n_parallel_sim_per_device:
                    pad_count = n_parallel_sim_per_device - chunk.shape[0]
                    pad_row = chunk[-1:, :]
                    chunk = jnp.vstack([chunk, jnp.repeat(pad_row, pad_count, axis=0)])
                batch = chunk

            device_batches.append(batch)

        sub_batch_params_shards = jnp.stack(device_batches, axis=0)

        result = parallel_sub_batch_simulate(sub_batch_params_shards)
        # jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)  # Ensure completion before measuring

        now = time.monotonic()
        if now - last_postfix_time > 0.25:
            pbar.set_postfix_str(_gpu_postfix_smi())
            last_postfix_time = now