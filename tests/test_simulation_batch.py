import os
import sys
import time
import json
import h5py
import multiprocessing
from multiprocessing import Queue
from datetime import datetime

# Keep JAX imports out of workers until after CUDA_VISIBLE_DEVICES is set.
import jax
import jax.numpy as jnp
import numpy as np

# Make repo importable (adjust if your tests folder layout differs)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import oscidyn
from oscidyn import const

# -------------------------
# Global configuration
# -------------------------

N_DUFFING = 20
DUFFING_COEFFICIENTS = jnp.linspace(-0.005, 0.03, N_DUFFING, dtype=oscidyn.const.DTYPE)

DRIVING_FREQUENCIES = jnp.linspace(0.1, 2.0, 200, dtype=oscidyn.const.DTYPE)
DRIVING_AMPLITUDES  = jnp.linspace(0.01, 1.0, 10, dtype=oscidyn.const.DTYPE)

# How many simulations to run in parallel per GPU (per vmap)
N_SIMULATIONS_IN_PARALLEL_PER_GPU = 2

# -------------------------
# Utilities
# -------------------------

def get_available_gpu_ids():
    """
    Returns a list of GPU IDs to use, respecting CUDA_VISIBLE_DEVICES if set.
    Falls back to JAX device discovery. Empty list if no GPUs.
    """
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        ids = [i.strip() for i in mask.split(",") if i.strip() != ""]
        return [int(i) for i in ids]
    try:
        gpus = jax.devices("gpu")
        return list(range(len(gpus)))
    except Exception:
        return []


def make_batches(arr: np.ndarray, batch_size: int):
    """
    Slice a 1D array into a list of 1D arrays (batches).
    """
    batches = []
    n = len(arr)
    for i in range(0, n, batch_size):
        batches.append(arr[i:i + batch_size])
    return batches


# -------------------------
# Worker
# -------------------------

def worker_process(gpu_id: int, task_queue: Queue, result_queue: Queue):
    """
    Worker pinned to a specific GPU.
    Pulls batched duffing arrays from task_queue, vmaps a single-sim function
    across the batch on that GPU, and pushes batched results to result_queue.
    """
    # Pin to GPU BEFORE importing JAX inside the worker
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now import JAX/oscidyn within this process
    import jax
    import jax.numpy as jnp
    import numpy as np
    import oscidyn

    def build_model(duffing_value):
        n_modes   = 1
        omega_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
        x_ref     = jnp.array(1.0, dtype=oscidyn.const.DTYPE)

        omega_0_hat = jnp.array([1.0], dtype=oscidyn.const.DTYPE)
        Q           = jnp.array([10.0], dtype=oscidyn.const.DTYPE)
        eta_hat     = jnp.array([0.005], dtype=oscidyn.const.DTYPE)

        alpha_hat = jnp.zeros((1, 1, 1), dtype=oscidyn.const.DTYPE)
        gamma_hat = jnp.zeros((1, 1, 1, 1), dtype=oscidyn.const.DTYPE)
        gamma_hat = gamma_hat.at[0, 0, 0, 0].set(duffing_value)
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
        return model

    def single_simulation(duffing_value):
        """
        Run ONE simulation (one frequency sweep) for a single duffing value.
        Returns array shaped [n_freq, n_amp] with the steady-state metric.
        """
        model = build_model(duffing_value)

        solver = oscidyn.FixedTimeSteadyStateSolver(
            max_steps=4096,
            n_time_steps=512,
            rtol=1e-4,
            atol=1e-6
        )

        result = oscidyn.vmap_safe_frequency_sweep(
            model=model,
            sweep_direction=oscidyn.SweepDirection.FORWARD,
            driving_frequencies=DRIVING_FREQUENCIES,
            driving_amplitudes=DRIVING_AMPLITUDES,
            solver=solver,
        )
        return result  # [n_freq, n_amp]

    single_simulation_jit = jax.jit(single_simulation)
    batched_simulations = jax.vmap(single_simulation_jit)

    # --- FIX: explicit loop with 'is None' to avoid NumPy == None comparison ---
    while True:
        batch_duffing = task_queue.get()
        if batch_duffing is None:
            break  # sentinel received

        try:
            # Convert to JAX array with correct dtype
            batch_duffing = jnp.asarray(batch_duffing, dtype=oscidyn.const.DTYPE)

            # Compute all sweeps in parallel on this GPU
            # Output: [batch_size, n_freq, n_amp]
            batch_result = batched_simulations(batch_duffing)

            # Materialize to host NumPy
            batch_result_np = np.array(batch_result)

            # Send the whole batch back to master
            result_queue.put({
                "duffing_values": np.array(batch_duffing),
                "results": batch_result_np
            })
        except Exception as e:
            result_queue.put({
                "error": True,
                "message": str(e)
            })

    return


# -------------------------
# Master / main
# -------------------------

if __name__ == "__main__":
    # ISO timestamp for file naming & metadata
    timestamp = datetime.now().isoformat(timespec='seconds')
    filename = f"tests/simulation_batch_{timestamp}.h5"

    ctx = multiprocessing.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    gpu_ids = get_available_gpu_ids()
    if not gpu_ids:
        raise RuntimeError("No GPUs detected. Set CUDA_VISIBLE_DEVICES or install CUDA/JAX correctly.")

    print(f"Detected {len(gpu_ids)} GPU(s): {gpu_ids}")

    workers = []
    for gpu in gpu_ids:
        p = ctx.Process(target=worker_process, args=(gpu, task_queue, result_queue))
        p.start()
        workers.append(p)

    num_gpus = len(gpu_ids)

    # Prepare batched tasks
    duffing_np = np.array(DUFFING_COEFFICIENTS)
    duffing_batches = make_batches(duffing_np, N_SIMULATIONS_IN_PARALLEL_PER_GPU)

    total_sims = len(duffing_np)
    total_batches = len(duffing_batches)

    # Enqueue all batches
    for b in duffing_batches:
        task_queue.put(b)

    # Send sentinel to each worker
    for _ in range(num_gpus):
        task_queue.put(None)

    start_time = time.time()

    # Open HDF5 once; master is sole writer
    with h5py.File(filename, "w") as h5f:
        settings = {
            "driving_frequencies": np.asarray(DRIVING_FREQUENCIES).tolist(),
            "driving_amplitudes": np.asarray(DRIVING_AMPLITUDES).tolist(),
            "sweep_direction": "FORWARD",
            "solver": {
                "name": "FixedTimeSteadyStateSolver",
                "max_steps": 4096,
                "n_time_steps": 512,
                "rtol": 1e-4,
                "atol": 1e-6
            },
            "n_simulations_in_parallel_per_gpu": int(N_SIMULATIONS_IN_PARALLEL_PER_GPU),
        }
        h5f.attrs["settings"] = json.dumps(settings)
        h5f.attrs["date"] = timestamp

        completed_sims = 0
        sim_index = 0

        while completed_sims < total_sims:
            payload = result_queue.get()

            if isinstance(payload, dict) and payload.get("error", False):
                raise RuntimeError(f"Worker error: {payload.get('message')}")

            duff_batch = payload["duffing_values"]          # [batch_size]
            results    = payload["results"]                 # [batch_size, n_freq, n_amp]

            for i in range(len(duff_batch)):
                sim_id = f"simulation_{sim_index:08d}"
                grp = h5f.create_group(sim_id)
                grp.create_dataset("maximum_steady_state_displacement", data=results[i])
                grp.attrs["dimensionless_parameters"] = np.array([duff_batch[i]])
                sim_index += 1
                completed_sims += 1

            h5f.flush()

    for p in workers:
        p.join()

    elapsed = time.time() - start_time
    sims_per_sec = total_sims / elapsed if elapsed > 0 else float('inf')
    print(f"Completed {total_sims} simulations in {elapsed:.2f} s "
          f"({sims_per_sec:.2f} sims/s) across {num_gpus} GPU(s).")
