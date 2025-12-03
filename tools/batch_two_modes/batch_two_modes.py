import jax
import numpy as np
from equinox import filter_jit
import oscidyn
import argparse

jax.config.update("jax_platform_name", "gpu")
N_FWHM = 10.0

SWEEP = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=50, max_steps=4096*3, verbose=False, throw=False, rtol=1e-4, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

TOTAL_SIMULATIONS = 100_000

Q_1_range = np.array([5.0, 20.0])
Q_2_range = np.array([5.0, 20.0])
omega_0_1_range = np.array([1.0, 1.0])
omega_0_2_range = np.array([1.0, 3.0])
gamma_1_multiplier_range = np.array([-1.0, 1.0])
gamma_2_multiplier_range = np.array([-1.0, 1.0])
eta_1_range = np.array([0.01, 1.0])
eta_2_range = np.array([0.01, 1.0])
modal_force_1_range = np.array([0.1, 1.0])
modal_force_2_range = np.array([0.0, 1.0])
alpha_range = np.array([0.0, 1.0])

# Generate random parameter sets
params = np.zeros((TOTAL_SIMULATIONS, 11))
params[:, 0] = np.random.uniform(Q_1_range[0], Q_1_range[1], TOTAL_SIMULATIONS)
params[:, 1] = np.random.uniform(Q_2_range[0], Q_2_range[1], TOTAL_SIMULATIONS)
params[:, 2] = np.random.uniform(omega_0_1_range[0], omega_0_1_range[1], TOTAL_SIMULATIONS)
params[:, 3] = np.random.uniform(omega_0_2_range[0], omega_0_2_range[1], TOTAL_SIMULATIONS)
params[:, 4] = np.random.uniform(eta_1_range[0], eta_1_range[1], TOTAL_SIMULATIONS)
params[:, 5] = np.random.uniform(eta_2_range[0], eta_2_range[1], TOTAL_SIMULATIONS)
params[:, 6] = np.random.uniform(modal_force_1_range[0], modal_force_1_range[1], TOTAL_SIMULATIONS)
params[:, 7] = np.random.uniform(modal_force_2_range[0], modal_force_2_range[1], TOTAL_SIMULATIONS)
params[:, 8] = np.random.uniform(alpha_range[0], alpha_range[1], TOTAL_SIMULATIONS)
params[:, 9] = np.random.uniform(gamma_1_multiplier_range[0], gamma_1_multiplier_range[1], TOTAL_SIMULATIONS)
params[:, 10] = np.random.uniform(gamma_2_multiplier_range[0], gamma_2_multiplier_range[1], TOTAL_SIMULATIONS)

print(f"Generated {TOTAL_SIMULATIONS} random parameter sets.")

# Q_values = np.linspace(1.1, 5.0, 5)  
# omega_0_values = np.linspace(1.0, 4.0, 5)
# gamma_values = np.linspace(-0.005, 0.005, 11)
# modal_force_values = np.linspace(0.1, 1.0, 5)

# # Generate all combinations of parameters for the two modes
# params = []
# for Q_1 in Q_values:
#     for Q_2 in Q_values:
#         for omega_0_1 in omega_0_values:
#             if omega_0_1 == 1.0:
#                 for omega_0_2 in omega_0_values:
#                     if omega_0_2 > omega_0_1:  # Ensure the second mode's resonance frequency is not lower
#                         for gamma_1 in gamma_values:
#                             for gamma_2 in gamma_values:
#                                 for modal_force_1 in modal_force_values:
#                                     for modal_force_2 in modal_force_values:
#                                         params.append([Q_1, Q_2, omega_0_1, omega_0_2, gamma_1, gamma_2, modal_force_1, modal_force_2])

# # Convert the list of parameters to a NumPy array
# params = np.array(params)
# print(f"Total parameter combinations: {params.shape[0]}")

# # Sort the parameters by Q_1 and then Q_2 in descending order
# params = params[np.lexsort((-params[:, 1], -params[:, 0]))]

@filter_jit
def gamma_activating_nonlinearity(Q, omega_0, f, eta):
    return (4 * eta * (1 + eta) / 3) * (omega_0**4 / (f**2 * Q**2))

@filter_jit
def simulate(params): # params: (n_params,)
    Q_1_val, Q_2_val, omega_0_1_val, omega_0_2_val, eta_1_val, eta_2_val, modal_force_1_val, modal_force_2_val, alpha_param, gamma_1_multiplier, gamma_2_multiplier = params
    
    Q_val = jnp.array([Q_1_val, Q_2_val])
    omega_0_val = jnp.array([omega_0_1_val, omega_0_2_val])
    
    alpha_val = jnp.zeros((2, 2, 2))
    gamma_val = jnp.zeros((2, 2, 2, 2))
    gamma_val = gamma_val.at[0, 0, 0, 0].set(gamma_activating_nonlinearity(Q_1_val, omega_0_1_val, modal_force_1_val, eta_1_val) * gamma_1_multiplier)
    gamma_val = gamma_val.at[1, 1, 1, 1].set(gamma_activating_nonlinearity(Q_2_val, omega_0_2_val, modal_force_2_val, eta_2_val) * gamma_2_multiplier)
    alpha_val = alpha_val.at[0, 0, 1].set(alpha_param)
    alpha_val = alpha_val.at[1, 0, 0].set(alpha_param)

    full_width_half_maxs = omega_0_val / Q_val

    model = oscidyn.BaseDuffingOscillator(Q=Q_val, omega_0=omega_0_val, alpha=alpha_val, gamma=gamma_val)
    drive_freq = jnp.linspace(
        jnp.maximum(omega_0_val[0] - N_FWHM * full_width_half_maxs[0], 0.1),
        jnp.maximum(omega_0_val[1] + N_FWHM * full_width_half_maxs[1], 0.1),
        200
    )
    #drive_amp = jnp.linspace(0.01, 1.0, 10)
    drive_amp = jnp.array([1.0])
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
                    sim_id = f"simulation_{sim_index:0{sim_width}d}"

                    sim_grp = grp.create_group(sim_id)

                    max_forward_sweep = jnp.max(batch_sweeps.sweeped_periodic_solutions['forward'][j])
                    max_backward_sweep = jnp.max(batch_sweeps.sweeped_periodic_solutions['backward'][j])

                    normalized_forward_sweep = np.asarray(batch_sweeps.sweeped_periodic_solutions['forward'][j]) / max_forward_sweep
                    normalized_backward_sweep = np.asarray(batch_sweeps.sweeped_periodic_solutions['backward'][j]) / max_backward_sweep

                    forward_sweep = sim_grp.create_dataset("forward_sweep", data=normalized_forward_sweep)
                    backward_sweep = sim_grp.create_dataset("backward_sweep", data=normalized_backward_sweep)
                    ds_x_max_total = sim_grp.create_dataset("unsweeped_total", data=np.asarray(batch_sweeps.periodic_solutions['max_x_total'][j]))
                    ds_x_max_modes = sim_grp.create_dataset("unsweeped_modes", data=np.asarray(batch_sweeps.periodic_solutions['max_x_modes'][j]))
                    #ds_x0 = sim_grp.create_dataset("x0", data=np.asarray(batch_sweeps.periodic_solutions['x0'][j]))
                    #ds_v0 = sim_grp.create_dataset("v0", data=np.asarray(batch_sweeps.periodic_solutions['v0'][j]))

                    forward_sweep.attrs['reference_displacement'] = np.float32(max_forward_sweep)
                    backward_sweep.attrs['reference_displacement'] = np.float32(max_backward_sweep)

                    f_omegas = np.asarray(batch_sweeps.f_omegas[j])
                    f_amps = np.asarray(batch_sweeps.f_amps[j])
                    Q = np.asarray(batch_sweeps.Q[j])
                    omega_0 = np.asarray(batch_sweeps.omega_0[j])
                    gamma = np.asarray(batch_sweeps.gamma[j])
                    modal_forces = np.asarray(batch_sweeps.modal_forces[j])
                    alpha = np.asarray(batch_sweeps.alpha[j])

                    success_rate = batch_sweeps.success_rate[j]

                    gamma_ndim_forward = max_forward_sweep**2 * gamma
                    gamma_ndim_backward = max_backward_sweep**2 * gamma

                    alpha_ndim_forward = max_forward_sweep * alpha
                    alpha_ndim_backward = max_backward_sweep * alpha

                    sim_grp.attrs['f_omegas'] = f_omegas
                    sim_grp.attrs['f_amps'] = f_amps
                    sim_grp.attrs['Q'] = Q
                    sim_grp.attrs['omega_0'] = omega_0
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
