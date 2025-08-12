# tests/batching/run_batch.py
import os
import sys
import time
import h5py
import subprocess
import multiprocessing as mp
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Make repo importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# -------------------------
# Helpers
# -------------------------

def get_available_gpu_ids():
    """Prefer CUDA_VISIBLE_DEVICES; fallback to nvidia-smi."""
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        return [int(i) for i in mask.split(",") if i.strip()]
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT
        ).decode().strip()
        return [int(x) for x in out.splitlines() if x.strip()]
    except Exception:
        return []

def load_tasks(h5_path):
    """
    Read work from the batch file.
    Returns:
        items: list of dict(group, Q, gamma, sweep_direction)
        driving_frequencies, driving_amplitudes: np.ndarray
    Skips groups that already have results.
    """
    items = []
    with h5py.File(h5_path, "r") as f:
        df = np.array(f.attrs["driving_frequencies"], dtype=float)
        da = np.array(f.attrs["driving_amplitudes"], dtype=float)
        for gname in sorted(f.keys()):
            grp = f[gname]
            has_res = "maximum_steady_state_displacement" in grp
            if not has_res:
                items.append({
                    "group": gname,
                    "Q": float(grp.attrs["Q"]),
                    "gamma": float(grp.attrs["gamma"]),
                    "sweep_direction": int(grp.attrs.get("sweep_direction", 1)),
                })
    return items, df, da

def chunk(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

# -------------------------
# Optional GPU monitor (simple)
# -------------------------

class GPUUtilMonitor(mp.Process):
    def __init__(self, gpu_ids, queue, interval=1.0):
        super().__init__(daemon=True)
        self.gpu_ids = list(gpu_ids)
        self.queue = queue
        self.interval = interval

    def run(self):
        if not self.gpu_ids:
            return
        id_arg = ",".join(map(str, self.gpu_ids))
        cmd = [
            "nvidia-smi",
            f"-i={id_arg}",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while True:
            try:
                out = subprocess.check_output(cmd).decode().strip()
                snap = {}
                for line in out.splitlines():
                    idx, util, mu, mt = map(str.strip, line.split(","))
                    snap[int(idx)] = {"util": int(util), "mu": float(mu), "mt": float(mt)}
                self.queue.put(snap)
            except Exception:
                pass
            time.sleep(self.interval)

# -------------------------
# Worker
# -------------------------

def worker(gpu_id, task_q, result_q, driving_frequencies, driving_amplitudes):
    # Pin to GPU and set JAX memory knobs BEFORE importing jax
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    
    import jax
    import jax.numpy as jnp
    import oscidyn
    import numpy as np

    df = jnp.asarray(driving_frequencies, dtype=oscidyn.const.DTYPE)
    da = jnp.asarray(driving_amplitudes,  dtype=oscidyn.const.DTYPE)

    # Shared model pieces
    n_modes   = 1
    omega_ref = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
    x_ref     = jnp.array(1.0, dtype=oscidyn.const.DTYPE)
    omega_0_hat = jnp.array([1.0],   dtype=oscidyn.const.DTYPE)
    eta_hat   = jnp.array([0.005], dtype=oscidyn.const.DTYPE)
    alpha0    = jnp.zeros((1, 1, 1), dtype=oscidyn.const.DTYPE)
    delta0    = jnp.zeros((1, 1, 1, 1, 1), dtype=oscidyn.const.DTYPE)

    def _build_model(Q_val, gamma_val):
        Q       = jnp.array([Q_val], dtype=oscidyn.const.DTYPE)
        gamma_hat   = jnp.zeros((1, 1, 1, 1), dtype=oscidyn.const.DTYPE)
        gamma_hat   = gamma_hat.at[0, 0, 0, 0].set(gamma_val)
        return oscidyn.NonlinearOscillator(
            n_modes=n_modes,
            Q=Q,
            eta_hat=eta_hat,
            alpha_hat=alpha0,
            gamma_hat=gamma_hat,
            delta_hat=delta0,
            omega_0_hat=omega_0_hat,
            omega_ref=omega_ref,
            x_ref=x_ref
        )

    def _run_simulation(Q_val, gamma_val, direction):
        model = _build_model(Q_val, gamma_val)
        solver = oscidyn.FixedTimeSteadyStateSolver(
            max_steps=4096, n_time_steps=512, rtol=1e-4, atol=1e-6, progress_bar=False
        )
        return oscidyn.vmap_safe_frequency_sweep(
            model=model,
            sweep_direction=direction,
            driving_frequencies=df,
            driving_amplitudes=da,
            solver=solver,
        )

    # Two separate JIT-compiled callables with fixed (Python) sweep direction
    run_forward  = jax.jit(lambda Q_val, gamma_val: _run_simulation(Q_val, gamma_val, oscidyn.SweepDirection.FORWARD))
    run_backward = jax.jit(lambda Q_val, gamma_val: _run_simulation(Q_val, gamma_val, oscidyn.SweepDirection.BACKWARD))

    while True:
        batch = task_q.get()
        if batch is None:
            break
        try:
            results = []
            for item in batch:
                g = item["group"]; Qv = item["Q"]; gv = item["gamma"]; sd = int(item["sweep_direction"])
                if sd >= 0:
                    arr = run_forward(Qv, gv)
                else:
                    arr = run_backward(Qv, gv)
                results.append((g, np.array(arr)))
            result_q.put(results)
        except Exception as e:
            result_q.put({"error": str(e), "groups": [it["group"] for it in batch]})


# -------------------------
# Main
# -------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Run simulations from a batch HDF5 file.")
    p.add_argument("batch_file", help="Path to HDF5 created by create_batch_file.py")
    p.add_argument("--batch-size", type=int, default=4, help="Simulations per GPU per step")
    p.add_argument("--monitor", action="store_true", help="Show live GPU utilization")
    args = p.parse_args()

    batch_file = args.batch_file
    if not os.path.exists(batch_file):
        raise FileNotFoundError(batch_file)

    tasks, df, da = load_tasks(batch_file)
    if not tasks:
        print("No pending simulations found. All done ✅")
        return

    gpus = get_available_gpu_ids()
    if not gpus:
        raise RuntimeError("No GPUs detected.")
    print(f"Detected {len(gpus)} GPU(s): {gpus}")

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()

    # Optional monitor
    mon_q = ctx.Queue()
    monitor = None
    if args.monitor:
        monitor = GPUUtilMonitor(gpus, mon_q, interval=1.0)
        monitor.start()

    # Start workers
    workers = []
    for gid in gpus:
        p = ctx.Process(target=worker, args=(gid, task_q, result_q, df, da))
        p.start()
        workers.append(p)

    # Build and enqueue batches; mark status = "running"
    batches = list(chunk(tasks, args.batch_size))
    with h5py.File(batch_file, "a") as f:
        for b in batches:
            for it in b:
                grp = f[it["group"]]
                grp.attrs["status"] = "running"
            task_q.put(b)
        for _ in gpus:
            task_q.put(None)
        f.flush()

    # Consume results
    start = time.time()
    total = len(tasks)
    done = 0
    with h5py.File(batch_file, "a") as f, tqdm(total=total, unit="sim", desc="Simulations") as pbar:
        while done < total:
            # light, optional monitor update
            if monitor is not None:
                try:
                    snap = mon_q.get_nowait()
                    parts = [f"gpu{gid} {s['util']:>2d}% {s['mu']:.0f}/{s['mt']:.0f} MiB"
                             for gid, s in sorted(snap.items())]
                    pbar.set_postfix_str(" | ".join(parts))
                except Exception:
                    pass

            payload = result_q.get()
            if isinstance(payload, dict) and "error" in payload:
                # mark errored groups so you can resume them later
                with h5py.File(batch_file, "a") as ferr:
                    for g in payload.get("groups", []):
                        if g in ferr:
                            ferr[g].attrs["status"] = "error"
                            ferr[g].attrs["error_message"] = payload["error"]
                raise RuntimeError(payload["error"])

            for gname, arr in payload:
                grp = f[gname]
                if "maximum_steady_state_displacement" in grp:
                    del grp["maximum_steady_state_displacement"]
                grp.create_dataset("maximum_steady_state_displacement", data=arr, compression="gzip")
                grp.attrs["status"] = "done"
                grp.attrs["completed_at"] = datetime.now().isoformat(timespec="seconds")
                done += 1
                pbar.update(1)
            f.flush()

    for p in workers:
        p.join()

    if monitor is not None:
        monitor.terminate()

    elapsed = time.time() - start
    print(f"Completed {total} simulations in {elapsed:.2f}s ({total/elapsed:.2f} sims/s).")

if __name__ == "__main__":
    main()
