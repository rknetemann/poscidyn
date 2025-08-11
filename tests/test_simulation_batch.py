import os
import sys
import time
import json
import h5py
import threading
import subprocess
import multiprocessing
from datetime import datetime
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DRIVING_FREQUENCIES = np.linspace(0.1, 2.0, 200)
DRIVING_AMPLITUDES  = np.linspace(0.01, 1.0, 10)

# -------------------------
# Utilities
# -------------------------

def get_available_gpu_ids():
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        ids = [i.strip() for i in mask.split(",") if i.strip() != ""]
        return [int(i) for i in ids]
    # Fallback to nvidia-smi to avoid importing JAX in the parent
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        return [int(x) for x in out.splitlines() if x.strip() != ""]
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
# GPU monitor (main process)
# -------------------------

class GPUUtilMonitor(threading.Thread):
    """
    Polls nvidia-smi periodically to collect GPU utilization and memory use.
    Also exposes last snapshot for tqdm postfix.
    """
    def __init__(self, gpu_ids, interval=1.0):
        super().__init__(daemon=True)
        self.gpu_ids = list(gpu_ids)
        self.interval = float(interval)
        self._stop = threading.Event()
        self.samples = []  # list of dicts {idx, gpu_util, mem_used, mem_total, ...}
        self._last = {}
        self._have_nvsmi = self._check_nvidia_smi()

    def _check_nvidia_smi(self):
        try:
            out = subprocess.run(["nvidia-smi", "-h"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out.returncode == 0
        except Exception:
            return False

    def run(self):
        if not self._have_nvsmi or not self.gpu_ids:
            return
        id_arg = ",".join(map(str, self.gpu_ids))
        cmd = [
            "nvidia-smi",
            f"-i={id_arg}",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits"
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
                snap_time = time.time()
                snapshot = {}
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) != 6:
                        continue
                    ts, idx, gpu_u, mem_u, mem_used, mem_tot = parts
                    idx = int(idx)
                    gpu_u = float(gpu_u)
                    mem_u = float(mem_u)
                    mem_used = float(mem_used)  # MiB
                    mem_tot = float(mem_tot)    # MiB
                    rec = {
                        "t": snap_time,
                        "idx": idx,
                        "gpu_util": gpu_u,
                        "mem_util": mem_u,
                        "mem_used": mem_used,
                        "mem_total": mem_tot
                    }
                    self.samples.append(rec)
                    snapshot[idx] = rec
                self._last = snapshot
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def last_snapshot(self):
        """Return dict {gpu_idx: {'util': int%, 'mem_used': MiB, 'mem_total': MiB}}."""
        return {
            k: {
                "util": int(v["gpu_util"]),
                "mem_used": v["mem_used"],
                "mem_total": v["mem_total"]
            }
            for k, v in self._last.items()
        }

    def summary(self):
        """Per-GPU summary."""
        if not self.samples:
            return {}
        summary = {}
        for gid in self.gpu_ids:
            s = [r for r in self.samples if r["idx"] == gid]
            if not s:
                continue
            gpu_utils = [r["gpu_util"] for r in s]
            mem_used = [r["mem_used"] for r in s]
            mem_tot = s[0]["mem_total"]
            summary[gid] = {
                "gpu_util_avg": float(np.mean(gpu_utils)),
                "gpu_util_max": float(np.max(gpu_utils)),
                "mem_used_max": float(np.max(mem_used)),
                "mem_total": float(mem_tot),
            }
        return summary

# -------------------------
# Worker
# -------------------------

def worker_process(gpu_id: int, task_queue, result_queue):
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
            atol=1e-6,
            progress_bar=False 
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

    while True:
        batch_duffing = task_queue.get()
        if batch_duffing is None:
            break  # sentinel received

        try:
            batch_duffing = jnp.asarray(batch_duffing, dtype=oscidyn.const.DTYPE)
            batch_result = batched_simulations(batch_duffing)  # [batch, n_freq, n_amp]
            batch_result_np = np.array(batch_result)
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
    filepath = os.environ.get("SIMULATIONS_OUTPUT_DIR", "tmp/output/")
    if not filepath.endswith(os.sep):
        filepath += os.sep
    if not os.path.exists(filepath):
        
        os.makedirs(filepath, exist_ok=True)
    n_simulations_in_parallel_per_gpu = int(os.environ.get("N_SIMULATIONS_IN_PARALLEL_PER_GPU", 4))
    n_duffing = int(os.environ.get("N_DUFFING", 20))

    duffing_coefficients = np.linspace(-0.005, 0.03, n_duffing)

    # ISO timestamp for file naming & metadata
    timestamp = datetime.now().isoformat(timespec='seconds').replace(":", "-")
    filename = f"{filepath}simulation_batch_{timestamp}.hdf5"

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
    duffing_np = np.array(duffing_coefficients)
    duffing_batches = make_batches(duffing_np, n_simulations_in_parallel_per_gpu)

    total_sims = len(duffing_np)
    total_batches = len(duffing_batches)

    # Enqueue all batches
    for b in duffing_batches:
        task_queue.put(b)

    # Send sentinel to each worker
    for _ in range(num_gpus):
        task_queue.put(None)

    start_time = time.time()

    # Start GPU monitoring (best-effort; harmless if nvidia-smi missing)
    monitor = GPUUtilMonitor(gpu_ids=gpu_ids, interval=1.0)
    monitor.start()

    # Open HDF5 once; master is sole writer
    with h5py.File(filename, "w") as h5f:
        h5f.attrs['n_simulations'] = n_duffing
        h5f.attrs['n_simulations_in_parallel_per_gpu'] = n_simulations_in_parallel_per_gpu
        h5f.attrs['driving_frequencies'] = DRIVING_FREQUENCIES
        h5f.attrs['driving_amplitudes'] = DRIVING_AMPLITUDES
        h5f.attrs['sweep_direction'] = "forward"
        h5f.attrs['solver'] = "FixedTimeSteadyStateSolver"
        h5f.attrs['max_steps'] = 4096
        h5f.attrs['n_time_steps'] = 512
        h5f.attrs['rtol'] = 1e-4
        h5f.attrs['atol'] = 1e-6
        h5f.attrs["created_date"] = timestamp

        completed_sims = 0
        sim_index = 0

        # NEW: tqdm progress bar
        with tqdm(total=total_sims, desc="Simulations", unit="sim") as pbar:
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
                    grp.attrs["gamma"] = np.array([duff_batch[i]])
                    sim_index += 1
                    completed_sims += 1
                    pbar.update(1)

                h5f.flush()

                # Live GPU util + memory on the bar
                snap = monitor.last_snapshot()
                if snap:
                    # Format: gpu0 73% 2500/24576 MiB (10%)
                    parts = []
                    for gid in sorted(snap):
                        util = snap[gid]["util"]
                        mu = snap[gid]["mem_used"]
                        mt = snap[gid]["mem_total"]
                        pct = (mu / mt * 100.0) if mt > 0 else 0.0
                        parts.append(f"gpu{gid} {util:>2d}% {mu:.0f}/{mt:.0f} MiB ({pct:.0f}%)")
                    pbar.set_postfix_str(" | ".join(parts))

    for p in workers:
        p.join()

    # Stop monitor and print summary
    monitor.stop()
    monitor.join(timeout=2.0)

    elapsed = time.time() - start_time
    sims_per_sec = total_sims / elapsed if elapsed > 0 else float('inf')
    print(f"Completed {total_sims} simulations in {elapsed:.2f} s "
          f"({sims_per_sec:.2f} sims/s) across {num_gpus} GPU(s).")

    # NEW: GPU utilization summary
    summary = monitor.summary()
    if summary:
        print("\nGPU utilization summary:")
        for gid in sorted(summary):
            s = summary[gid]
            mem_pct = (s["mem_used_max"] / s["mem_total"] * 100.0) if s["mem_total"] > 0 else 0.0
            print(
                f" - GPU {gid}: avg util {s['gpu_util_avg']:.1f}%, "
                f"max util {s['gpu_util_max']:.1f}%, "
                f"max mem {s['mem_used_max']:.0f}/{s['mem_total']:.0f} MiB ({mem_pct:.1f}%)"
            )
    else:
        print("\nGPU monitoring disabled or no samples collected (is 'nvidia-smi' available?).")
