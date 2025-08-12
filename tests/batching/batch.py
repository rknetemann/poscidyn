import os
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"  
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "false"  # Disable preallocation to avoid OOM errors
import jax
import jax.numpy as jnp
import sys
import time
import math
from tqdm import tqdm
import subprocess
import time
import numpy as np
import h5py

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.gpu_monitor import GpuMonitor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn
import argparse

driving_frequencies = jnp.linspace(0.1, 2.0, 200)
driving_amplitudes = jnp.linspace(0.01, 1.0, 10)

Q = jnp.linspace(1.1, 10.0, 50)  
gamma = jnp.linspace(-0.001, 0.001, 50)  
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
    
simulate_sub_batch = jax.vmap(simulate) # input args: (n_parallel_sim, n_params)

params = jnp.column_stack((Q, gamma, sweep_direction))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch simulate nonlinear oscillator frequency sweeps")
    parser.add_argument(
        "--n_parallel_sim",
        type=int,
        default=2,
        nargs="?",
        help="Amount of simulations to run in parallel (default: 2)"
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default=None,
        help="Optional output file name for results"
    )
    parser.add_argument(
        "--file_overwrite",
        action="store_true",
        help="Overwrite output file if it exists"
    )

    args = parser.parse_args()

    n_parallel_sim = args.n_parallel_sim
    file_name = args.file_name
    file_overwrite = args.file_overwrite

    if file_name:
        dir_path = os.path.dirname(file_name)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(file_name) and not file_overwrite:
            print(f"Error: File '{file_name}' already exists. Use --file_overwrite to overwrite file.", file=sys.stderr)
            sys.exit(1)
    
    n_sim = len(Q)
    n_sub_batches = math.ceil(n_sim / (n_parallel_sim)) # Example: 1001 simulations, 10 simulations in parallel -> 101 sub-batches
    
    print(f"Total simulations: {n_sim}, Parallel simulations: {n_parallel_sim}, Sub-batches: {n_sub_batches}")
    
    with h5py.File(file_name, 'w') as f:
        with GpuMonitor(interval=0.5) as gm:
            pbar = tqdm(range(n_sub_batches), desc="Simulating", unit="batch", dynamic_ncols=True)
            start_time = time.time()
            for i in pbar:
                start_idx = i * n_parallel_sim
                end_idx = min(start_idx + n_parallel_sim, n_sim)

                batch_params = params[start_idx:end_idx]
                t0 = time.time()
                batch_sweeps = simulate_sub_batch(batch_params)
                elapsed = time.time() - t0

                n_in_batch = batch_params.shape[0]
                secs_per_sim = elapsed / n_in_batch

                postfix_parts = [f"{secs_per_sim:.2f}s/sim"]
                gpu_line = gm.summary()
                if gpu_line:
                    postfix_parts.append(gpu_line)

                pbar.set_postfix_str("   ".join(postfix_parts))

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    elapsed = time.time() - start_time
    print(f"Simulations per second: {n_sim / elapsed:.2f}")