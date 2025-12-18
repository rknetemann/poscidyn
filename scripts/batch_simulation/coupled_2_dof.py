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

jax.config.update("jax_platform_name", "gpu")

SWEEP = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=50, max_steps=5*4096, verbose=False, throw=False, rtol=1e-4, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

TOTAL_SIMULATIONS = 10_000
ETA = 0.2

Q_1_range = np.array([5.0, 100.0])
Q_2_range = np.array([5.0, 100.0])
omega_0_multiplier_range = np.array([0.9, 1.1])
gamma_multiplier_range = np.array([0.9, 1.1])
modal_force_1_range = np.array([0.1, 1.0])
modal_force_2_range = np.array([0.0, 1.0])

params = np.zeros((TOTAL_SIMULATIONS, 10))
params[:, 0] = np.random.uniform(Q_1_range[0], Q_1_range[1], TOTAL_SIMULATIONS)
params[:, 1] = np.random.uniform(Q_2_range[0], Q_2_range[1], TOTAL_SIMULATIONS)
params[:, 2] = np.random.uniform(omega_0_multiplier_range[0], omega_0_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 3] = np.random.uniform(omega_0_multiplier_range[0], omega_0_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 4] = np.random.uniform(gamma_multiplier_range[0], gamma_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 5] = np.random.uniform(gamma_multiplier_range[0], gamma_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 6] = np.random.uniform(gamma_multiplier_range[0], gamma_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 7] = np.random.uniform(gamma_multiplier_range[0], gamma_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 8] = np.random.uniform(modal_force_1_range[0], modal_force_1_range[1], TOTAL_SIMULATIONS)
params[:, 9] = np.random.uniform(modal_force_2_range[0], modal_force_2_range[1], TOTAL_SIMULATIONS)

@filter_jit
def F_max (eta, omega_0, Q, gamma):
    return jnp.sqrt(4 * omega_0**6 / (3 * gamma * Q**2) * (eta + 1 / (2*Q**2)) * (1 + eta + 1 / (4 * Q **2)))

@filter_jit
def simulate(params): # params: (n_params,)
    Q_1_val, Q_2_val, omega_0_1_mult, omega_0_2_mult, gamma_1_mult, gamma_2_mult, gamma_3_mult, gamma_4_mult, modal_force_1_val, modal_force_2_val = params
    
    Q_val = jnp.array([Q_1_val, Q_2_val])
    omega_0_val = jnp.array([1.0 * omega_0_1_mult, 1.73 * omega_0_2_mult])
    alpha = jnp.zeros((2,2,2))
    gamma = jnp.zeros((2,2,2,2))
    gamma = gamma.at[0,0,0,0].set(2.55 * gamma_1_mult)
    gamma = gamma.at[0,0,1,1].set(8.61 * gamma_2_mult)
    gamma = gamma.at[1,1,1,1].set(18.7 * gamma_3_mult)
    gamma = gamma.at[1,0,0,1].set(8.57 * gamma_4_mult)

    model = oscidyn.BaseDuffingOscillator(Q=Q_val, omega_0=omega_0_val, alpha=alpha, gamma=gamma)
    drive_freq = jnp.linspace(0.75, 3.0, 601)
    drive_amp = jnp.linspace(0.1, 1.0, 10) * F_max(ETA, omega_0_val[0], Q_val[0], gamma[0,0,0,0])
    excitation = oscidyn.OneToneExcitation(drive_frequencies=drive_freq, drive_amplitudes=drive_amp, modal_forces=jnp.array([modal_force_1_val, modal_force_2_val]))

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

                    # Post-processing and normalization
                    f_omegas = batch_sweeps.f_omegas[j]

                    x_total = batch_sweeps.periodic_solutions['max_x_total'][j]
                    x_modes = batch_sweeps.periodic_solutions['max_x_modes'][j]

                    x_forward  = batch_sweeps.sweeped_periodic_solutions['forward'][j]
                    x_backward = batch_sweeps.sweeped_periodic_solutions['backward'][j]

                    omega_0_0 = batch_sweeps.omega_0[j][0]
                    
                    ref_idx = jnp.argmin(jnp.abs(f_omegas - 0.9 * omega_0_0))
                    
                    omega_ref = f_omegas[ref_idx]
                    x_ref_forward = x_forward[ref_idx, :]
                    x_ref_backward = x_backward[ref_idx, :]

                    normalized_x_forward = np.asarray(x_forward) / x_ref_forward
                    normalized_x_backward = np.asarray(x_backward) / x_ref_backward
                    normalized_omega = np.asarray(f_omegas) / omega_ref

                    norm_alpha_forward = (x_ref_forward / omega_ref**2)
                    norm_alpha_backward = (x_ref_backward / omega_ref**2)
                    norm_gamma_forward = (x_ref_forward**2 / omega_ref**2)
                    norm_gamma_backward = (x_ref_backward**2 / omega_ref**2)
                    
                    alpha_ndim_forward  = norm_alpha_forward[:, None, None, None] * alpha[None, :, :, :]         # (n_amp,2,2,2)
                    alpha_ndim_backward = norm_alpha_backward[:, None, None, None] * alpha[None, :, :, :]
                    gamma_ndim_forward  = norm_gamma_forward[:, None, None, None, None] * gamma[None, :, :, :, :] 
                    gamma_ndim_backward = norm_gamma_backward[:, None, None, None, None] * gamma[None, :, :, :, :]  

                    f_omegas = np.asarray(batch_sweeps.f_omegas[j])
                    f_amps = np.asarray(batch_sweeps.f_amps[j])
                    Q = np.asarray(batch_sweeps.Q[j])
                    omega_0 = np.asarray(batch_sweeps.omega_0[j])
                    gamma = np.asarray(batch_sweeps.gamma[j])
                    modal_forces = np.asarray(batch_sweeps.modal_forces[j])
                    alpha = np.asarray(batch_sweeps.alpha[j])

                    success_rate = batch_sweeps.success_rate[j]

                    # Storing data in HDF5
                    sim_id = f"simulation_{sim_index:0{sim_width}d}"
                    sim_grp = grp.create_group(sim_id)

                    forward_sweep = sim_grp.create_dataset("forward_sweep", data=normalized_x_forward)
                    backward_sweep = sim_grp.create_dataset("backward_sweep", data=normalized_x_backward)
                    ds_x_max_total = sim_grp.create_dataset("unsweeped_total", data=np.asarray(x_total))
                    ds_x_max_modes = sim_grp.create_dataset("unsweeped_modes", data=np.asarray(x_modes))

                    forward_sweep.attrs['reference_displacement'] = x_ref_forward
                    backward_sweep.attrs['reference_displacement'] = x_ref_backward
                    forward_sweep.attrs['reference_frequency'] = omega_ref
                    backward_sweep.attrs['reference_frequency'] = omega_ref

                    sim_grp.attrs['f_omegas'] = f_omegas
                    sim_grp.attrs['f_amps'] = f_amps
                    sim_grp.attrs['Q'] = Q
                    sim_grp.attrs['omega_0'] = omega_0
                    sim_grp.attrs['gamma'] = gamma
                    sim_grp.attrs['alpha'] = alpha
                    sim_grp.attrs['modal_forces'] = modal_forces
                    sim_grp.attrs['success_rate'] = success_rate

                    forward_sweep.attrs['gamma_ndim'] = gamma_ndim_forward
                    backward_sweep.attrs['gamma_ndim'] = gamma_ndim_backward
                    forward_sweep.attrs['alpha_ndim'] = alpha_ndim_forward
                    backward_sweep.attrs['alpha_ndim'] = alpha_ndim_backward


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
