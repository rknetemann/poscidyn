import os
import jax
import jax.numpy as jnp
import sys
import time
import math
from tqdm import tqdm
import time
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

params = jnp.column_stack((Q, gamma, sweep_direction))
params = params[jnp.argsort(params[:, 0])] # sort by Q values (column 0)

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
        "--n_tasks",
        type=int,
        default=2,
        help="Number of tasks to divide the batches into (default: 2)"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Which task to run (default: 0)"
    )
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

    n_tasks = args.n_tasks
    task_id = args.task_id
    n_parallel_sim = args.n_parallel_sim
    file_name = args.file_name
    file_overwrite = args.file_overwrite

    if task_id != "":
        task_id_formatted = f"_{task_id}"
    else:
        task_id_formatted = ""

    if file_name:
        dir_path = os.path.dirname(file_name)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(file_name) and not file_overwrite:
            print(f"Error: File '{file_name}' already exists. Use --file_overwrite to overwrite file.", file=sys.stderr)
            sys.exit(1)
    else:
        datetime = time.strftime("%Y-%m-%d_%H:%M:%S")
        file_name = f"batch_{datetime}{task_id_formatted}.hdf5"

    task_param = [params[i::n_tasks] for i in range(n_tasks)][task_id]

    n_sim = len(task_param)
    n_batches = math.ceil(n_sim / (n_parallel_sim)) # Example: 1001 simulations, 10 simulations in parallel -> 101 sub-batches
    
    print(f"Total simulations: {n_sim}, Parallel simulations: {n_parallel_sim}, Batches: {n_batches}")
    
    with h5py.File(file_name, 'w') as hdf5:
        hdf5.create_dataset('drving_frequencies', data=driving_frequencies)
        hdf5.create_dataset('driving_amplitudes', data=driving_amplitudes)
        hdf5.create_dataset('params', data=task_param)
        hdf5.attrs['task_id'] = task_id
        hdf5.attrs['n_simulations'] = n_sim
        hdf5.attrs['n_parallel_simulations'] = n_parallel_sim
        hdf5.attrs['n_batches'] = n_batches
        hdf5.attrs['started_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        hdf5.attrs['completed_at'] = ""

        with GpuMonitor(interval=0.5) as gm:
            pbar = tqdm(range(n_batches), desc="Simulating", unit="batch", dynamic_ncols=True)
            start_time = time.time()
            for i in pbar:
                start_idx = i * n_parallel_sim
                end_idx = min(start_idx + n_parallel_sim, n_sim)

                batch_params = task_param[start_idx:end_idx]
                t0 = time.time()
                batch_sweeps = simulate_sub_batch(batch_params)
                elapsed = time.time() - t0

                for j, batch_sweep in enumerate(batch_sweeps): # batch_sweep: (n_driving_frequencies * n_driving_amplitudes)
                    sim_index = start_idx + j
                    sim_width = len(str(n_sim))
                    sim_id = f"simulation_{sim_index:0{sim_width-1}d}"
                    grp = hdf5.create_group(sim_id)
                    
                    grp.attrs['Q'] = batch_params[j, 0]
                    grp.attrs['gamma'] = batch_params[j, 1]
                    grp.attrs['sweep_direction'] = batch_params[j, 2]
                    
                    grp.create_dataset('max_steady_state_displacement', data=batch_sweep)

                n_in_batch = batch_params.shape[0]
                secs_per_sim = elapsed / n_in_batch

                postfix_parts = [f"{secs_per_sim:.2f}s/sim"]
                gpu_line = gm.summary()
                if gpu_line:
                    postfix_parts.append(gpu_line)

                pbar.set_postfix_str("   ".join(postfix_parts))
                hdf5.attrs['max_gpu_usage'] = gm.max_summary() if gm._max_summary else ""
        
            hdf5.attrs['completed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hdf5.attrs['elapsed_time'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            hdf5.attrs['n_simulations_per_second'] = n_sim / (time.time() - start_time)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    elapsed = time.time() - start_time
    print(f"Simulations per second: {n_sim / elapsed:.2f}")