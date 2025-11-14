import os
import jax
import jax.numpy as jnp
import sys
import time
import math
from tqdm import tqdm
import time
import h5py
import numpy as np
from equinox import filter_jit

from utils.gpu_monitor import GpuMonitor
import oscidyn
import argparse

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
N_FWHM = 4.0

SWEEP = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=200, max_steps=4096*3, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

Q_values = np.linspace(10.0, 20.0, 5)  
omega_0_values = np.linspace(0.9, 4.0, 5)
gamma_values = np.linspace(0.0, 0.005, 5)
modal_force_values = np.linspace(0.1, 1.0, 5)

# Generate all combinations of parameters for the two modes
params = []
for Q_1 in Q_values:
    for Q_2 in Q_values:
        for omega_0_1 in omega_0_values:
            for omega_0_2 in omega_0_values:
                if omega_0_2 > omega_0_1:  # Ensure the second mode's resonance frequency is not lower
                    for gamma_1 in gamma_values:
                        for gamma_2 in gamma_values:
                            for modal_force_1 in modal_force_values:
                                for modal_force_2 in modal_force_values:
                                    if modal_force_2 <= modal_force_1:  # Ensure the second mode's modal force is not greater
                                        params.append([Q_1, Q_2, omega_0_1, omega_0_2, gamma_1, gamma_2])

# Convert the list of parameters to a NumPy array
params = np.array(params)

# Sort the parameters by Q_1 and then Q_2 in descending order
params = params[np.lexsort((-params[:, 1], -params[:, 0]))]

@filter_jit
def simulate(params): # params: (n_params,)
    Q_1_val, Q_2_val, omega_0_1_val, omega_0_2_val, gamma_1_val, gamma_2_val = params
    
    Q_val = jnp.array([Q_1_val, Q_2_val])
    omega_0_val = jnp.array([omega_0_1_val, omega_0_2_val])
    
    alpha_val = jnp.zeros((2, 2, 2))
    gamma_val = jnp.zeros((2, 2, 2, 2))
    gamma_val = gamma_val.at[0, 0, 0, 0].set(gamma_1_val)
    gamma_val = gamma_val.at[1, 1, 1, 1].set(gamma_2_val)

    full_width_half_maxs = omega_0_val / Q_val

    model = oscidyn.BaseDuffingOscillator(Q=Q_val, omega_0=omega_0_val, alpha=alpha_val, gamma=gamma_val)
    drive_freq = jnp.linspace(
        jnp.maximum(omega_0_val[0] - N_FWHM * full_width_half_maxs[0], 0.1),
        jnp.maximum(omega_0_val[1] + N_FWHM * full_width_half_maxs[1], 0.1),
        300
    )
    drive_amp = jnp.linspace(0.01, 1.0, 10)
    excitation = oscidyn.OneToneExcitation(drive_frequencies=drive_freq, drive_amplitudes=drive_amp, modal_forces=np.array([1.0, 0.5]))

    return oscidyn.frequency_sweep(
        model=model,
        sweeper=SWEEP,
        excitor=excitation,
        solver=SOLVER,
        multistarter=MULTISTART,
        precision=PRECISION,
    )
    
simulate_sub_batch = jax.vmap(simulate) # input args: (n_parallel_sim, n_params)

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
        default=4,
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
        hdf5.create_dataset('params', data=np.asarray(task_param))
        hdf5.attrs['task_id'] = task_id
        hdf5.attrs['n_simulations'] = n_sim
        hdf5.attrs['n_parallel_simulations'] = n_parallel_sim
        hdf5.attrs['n_batches'] = n_batches
        hdf5.attrs['started_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
        hdf5.attrs['completed_at'] = ""
        
        grp = hdf5.create_group('simulations')

        with GpuMonitor(interval=0.5) as gm:
            pbar = tqdm(range(n_batches), desc="Simulating", unit="batch", dynamic_ncols=True)
            start_time = time.time()
            for i in pbar:
                start_idx = i * n_parallel_sim
                end_idx = min(start_idx + n_parallel_sim, n_sim)

                # if i==0:
                #     jax.profiler.start_trace(f"profiling/sub_batch_{n_parallel_sim}_in_parallel")

                batch_params = task_param[start_idx:end_idx]
                t0 = time.time()
                batch_sweeps = simulate_sub_batch(batch_params)
                elapsed = time.time() - t0
                
                # if i==0:
                #     jax.profiler.stop_trace()

                n_in_batch = batch_params.shape[0]
                sim_width = len(str(n_sim - 1)) if n_sim > 1 else 1
                for j in range(n_in_batch):
                    sim_index = start_idx + j
                    sim_id = f"simulation_{sim_index:0{sim_width}d}"

                    sim_grp = grp.create_group(sim_id)

                    sweeped_periodic_solutions = sim_grp.create_dataset("sweeped_periodic_solutions", data=np.asarray(batch_sweeps.sweeped_periodic_solutions['forward'][j]))
                    ds_x_max_total = sim_grp.create_dataset("x_max_total", data=np.asarray(batch_sweeps.periodic_solutions['max_x_total'][j]))
                    ds_x_max_modes = sim_grp.create_dataset("x_max_modes", data=np.asarray(batch_sweeps.periodic_solutions['max_x_modes'][j]))
                    #ds_x0 = sim_grp.create_dataset("x0", data=np.asarray(batch_sweeps.periodic_solutions['x0'][j]))
                    #ds_v0 = sim_grp.create_dataset("v0", data=np.asarray(batch_sweeps.periodic_solutions['v0'][j]))

                    sim_grp.attrs['f_omegas'] = np.asarray(batch_sweeps.f_omegas[j])
                    sim_grp.attrs['f_amps'] = np.asarray(batch_sweeps.f_amps[j])
                    sim_grp.attrs['Q'] = np.asarray(batch_sweeps.Q[j])
                    sim_grp.attrs['omega_0'] = np.asarray(batch_sweeps.omega_0[j])
                    sim_grp.attrs['gamma'] = np.asarray(batch_sweeps.gamma[j])

                secs_per_sim = elapsed / max(n_in_batch, 1)

                postfix_parts = [f"{secs_per_sim:.2f}s/sim"]
                gpu_line = gm.summary()
                if gpu_line:
                    postfix_parts.append(gpu_line)

                pbar.set_postfix_str("   ".join(postfix_parts))
                hdf5.attrs['max_gpu_usage'] = gm.max_summary() if gm._max_summary else ""
                
                # if i == 10:
                #     break # Early exit for debugging
        
            hdf5.attrs['completed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            hdf5.attrs['elapsed_time'] = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            hdf5.attrs['n_simulations_per_second'] = n_sim / (time.time() - start_time)

    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    elapsed = time.time() - start_time
    print(f"Simulations per second: {n_sim / elapsed:.2f}")