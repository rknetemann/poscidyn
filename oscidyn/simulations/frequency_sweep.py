# frequency_sweep.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple
import numpy as np

from oscidyn.models import AbstractModel
from oscidyn.solver import AbstractSolver,SteadyStateSolver, FixedTimeSteadyStateSolver, FixedTimeSolver
from oscidyn.constants import SweepDirection
from oscidyn.results import FrequencySweepResult
import time
from mpl_toolkits.mplot3d import Axes3D
import oscidyn.constants as const

# TO DO: Improve steady state amplitude calculation
# ASSUMPTION: The steady state is already reached in the time response
@jax.jit
def _get_steady_state_part(
    drive_freq: jax.Array, # (n_sim,)
    time: jax.Array, # (n_sim, n_time_steps)
    disp: jax.Array, # (n_sim, n_time_steps, n_modes)
    vel: jax.Array, # (n_sim, n_time_steps, n_modes)
):
    # shapes
    n_sim, n_time_steps, n_modes = disp.shape

    # time step for each simulation
    dt = time[:, 1] - time[:, 0]  # (n_sim,)

    # number of samples per period
    samp_per_per = jnp.round((2.0 * jnp.pi / drive_freq) / dt).astype(int)  # (n_sim,)

    # we keep the last N_PERIODS_TO_RETAIN periods
    start_idx = n_time_steps - const.N_PERIODS_TO_RETAIN * samp_per_per  # (n_sim,)

    # build mask
    time_idx = jnp.arange(n_time_steps)[None, :]       # (1, n_time_steps)
    in_window = time_idx >= start_idx[:, None]         # (n_sim, n_time_steps)

    # also restrict to time > 0 (settling time)
    in_window = in_window & (time > 0)            # (n_sim, n_time_steps)

    # expand mask for disp/vel dims
    in_window3 = in_window[..., None]                  # (n_sim, n_time_steps, 1)

    # mask out entire steady‐state slice
    ss_time = jnp.where(in_window, time, 0.0)                 # (n_sim, n_time_steps)
    ss_disp = jnp.where(in_window3, disp, 0.0)            # (n_sim, n_time_steps, n_modes)
    ss_vel  = jnp.where(in_window3, vel, 0.0)            # (n_sim, n_time_steps, n_modes)

    return ss_time, ss_disp, ss_vel
    
# ASSUMPTION: Initial velocity does not have to be tested for finding branches
def _explore_branches(
    model: AbstractModel,
    drive_freq: jax.Array, # (n_freq,)
    drive_amp:  jax.Array, # (n_amp,)
    solver: AbstractSolver,
    sweep_direction: SweepDirection,
) -> Tuple[jax.Array, jax.Array]:
    
    n_simulations = const.N_COARSE_DRIVING_FREQUENCIES * const.N_COARSE_DRIVING_AMPLITUDES \
    * const.N_COARSE_INITIAL_DISPLACEMENTS * const.N_COARSE_INITIAL_VELOCITIES
    n_simulations_formatted = f"{n_simulations:,}".replace(",", ".")
    print("\nBasin exploration:")
    print(f"-> running {n_simulations_formatted} simulations in parallel...")
    start_time = time.time()

    coarse_drive_freq = jnp.linspace(
        jnp.min(drive_freq), jnp.max(drive_freq), const.N_COARSE_DRIVING_FREQUENCIES
    ) # (N_COARSE_DRIVING_FREQUENCIES,)

    coarse_drive_amp = jnp.linspace(
        jnp.min(drive_amp), jnp.max(drive_amp), const.N_COARSE_DRIVING_AMPLITUDES
    ) # (N_COARSE_DRIVING_AMPLITUDES,)

    max_abs_displacement = 10.0 # TO DO: Determine the max amplitude based on the model or a fixed value
    coarse_init_disp = jnp.linspace(
        -max_abs_displacement, max_abs_displacement, const.N_COARSE_INITIAL_DISPLACEMENTS
    ) # (N_COARSE_INITIAL_DISPLACEMENTS,)
    
    max_abs_velocity = 10.0 # TO DO: Determine the max velocity based on the model or a fixed value
    coarse_init_vel = jnp.linspace(
        -max_abs_velocity, max_abs_velocity, const.N_COARSE_INITIAL_VELOCITIES
    ) # (N_COARSE_INITIAL_VELOCITIES,)

    coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh, coarse_init_vel_mesh = jnp.meshgrid(
        coarse_drive_freq, coarse_drive_amp, coarse_init_disp, coarse_init_vel, indexing="ij"
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES)

    coarse_drive_freq_flat = coarse_drive_freq_mesh.ravel()
    coarse_drive_amp_flat = coarse_drive_amp_mesh.ravel()
    coarse_init_disp_flat = coarse_init_disp_mesh.ravel()
    coarse_init_vel_flat = coarse_init_vel_mesh.ravel()
    # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS)
    
    @jax.jit
    def solve_case(drive_freq, drive_amp, init_disp, init_vel):
        init_disp = jnp.full((model.n_modes,), init_disp)
        init_vel = jnp.full((model.n_modes,), init_vel)
        init_cond = jnp.concatenate([init_disp, init_vel])

        return solver(model, drive_freq, drive_amp, init_cond, const.ResponseType.FrequencyResponse)

    sol = jax.vmap(solve_case)(coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat)

    if isinstance(solver, SteadyStateSolver):
        # SteadyStateSolver only returns the last window (which is the steady state part)
        ts, ys = sol

        disp = ys[..., :model.n_modes] # (n_sim, n_windows, n_steps, n_modes)
        vel  = ys[..., model.n_modes:] # (n_sim, n_windows, n_steps, n_modes)

        sample_mask = jnp.any(jnp.abs(ys) > 0, axis=-1) # (n_sim, n_windows, n_steps)
        sample_mask = sample_mask[..., None] # broadcast over modes

        # TO DO: Implement it for SteadyStateSolver
        raise NotImplementedError("SteadyStateSolver is not yet implemented for frequency sweep.")
    elif isinstance(solver, FixedTimeSolver):
        # FixedTimeSolver returns the entire time response, not just the steady state part
        # TO DO: Implement it for FixedTimeSolver
        raise NotImplementedError("FixedTimeSolver is not yet implemented for frequency sweep.")
    elif isinstance(solver, FixedTimeSteadyStateSolver):
        # FixedTimeSteadyStateSolver only returns the steady state part
        ts, ys = sol

        ss_time_flat = ts # (n_sim, n_time_steps)
        ss_disp_flat = ys[..., :model.n_modes] # (n_sim, n_time_steps, n_modes)
        ss_vel_flat = ys[..., model.n_modes:] # (n_sim, n_time_steps, n_modes)

        idx_max_disp = jnp.argmax(jnp.abs(ss_disp_flat), axis=1) # (n_sim, n_modes)
        t_max_disp = jnp.take_along_axis(ss_time_flat[:, :, None], idx_max_disp[:, None, :], axis=1).squeeze(1) # (n_sim, n_modes)
        q_max_disp = jnp.abs(jnp.take_along_axis(ss_disp_flat, idx_max_disp[:, None, :], axis=1).squeeze(1)) # (n_sim, n_modes)
        v_max_disp = jnp.abs(jnp.take_along_axis(ss_vel_flat, idx_max_disp[:, None, :], axis=1).squeeze(1)) # (n_sim, n_modes)
        y_max_disp = jnp.concatenate([q_max_disp, v_max_disp], axis=-1) # (n_sim, n_modes * 2)

        idx_max_vel = jnp.argmax(jnp.abs(ss_vel_flat), axis=1) # (n_sim, n_modes)
        t_max_vel = jnp.take_along_axis(ss_time_flat[:, :, None], idx_max_vel[:, None, :], axis=1).squeeze(1) # (n_sim, n_modes)
        q_max_vel = jnp.abs(jnp.take_along_axis(ss_disp_flat, idx_max_vel[:, None, :], axis=1).squeeze(1))  # (n_sim, n_modes)
        v_max_vel = jnp.abs(jnp.take_along_axis(ss_vel_flat, idx_max_vel[:, None, :], axis=1).squeeze(1))  # (n_sim, n_modes)
        y_max_vel = jnp.concatenate([q_max_vel, v_max_vel], axis=-1) # (n_sim, n_modes * 2)

    t_max_disp = t_max_disp.reshape(
        const.N_COARSE_DRIVING_FREQUENCIES, 
        const.N_COARSE_DRIVING_AMPLITUDES, 
        const.N_COARSE_INITIAL_DISPLACEMENTS, 
        const.N_COARSE_INITIAL_VELOCITIES,
        model.n_modes
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)

    y_max_disp = y_max_disp.reshape(
        const.N_COARSE_DRIVING_FREQUENCIES, 
        const.N_COARSE_DRIVING_AMPLITUDES, 
        const.N_COARSE_INITIAL_DISPLACEMENTS, 
        const.N_COARSE_INITIAL_VELOCITIES,
        model.n_modes * 2
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)

    freq_idx = 150

    import matplotlib.pyplot as plt

    # flatten the coarse‐grid for plotting
    freq_vals = coarse_drive_freq_mesh.ravel()
    amp_vals  = coarse_drive_amp_mesh.ravel()
    disp_vals = jnp.abs(y_max_disp[..., 0]).ravel()

    plt.figure()
    # background scatter in gray
    sc = plt.scatter(
        freq_vals,
        disp_vals,
        c=amp_vals,
        cmap='Greys',
        vmin=amp_vals.min(),
        vmax=amp_vals.max()
    )
    plt.colorbar(sc, label='Driving amplitude')

    # highlight points at the chosen frequency
    freq_target = freq_vals[freq_idx]
    mask = freq_vals == freq_target

    plt.xlabel('Driving frequency')
    plt.ylabel('Max steady‐state displacement (mode 0)')
    plt.tight_layout()
    plt.show()

    t_max_vel = t_max_vel.reshape(
        const.N_COARSE_DRIVING_FREQUENCIES, 
        const.N_COARSE_DRIVING_AMPLITUDES, 
        const.N_COARSE_INITIAL_DISPLACEMENTS, 
        const.N_COARSE_INITIAL_VELOCITIES,
        model.n_modes
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)  

    y_max_vel = y_max_vel.reshape(
        const.N_COARSE_DRIVING_FREQUENCIES, 
        const.N_COARSE_DRIVING_AMPLITUDES, 
        const.N_COARSE_INITIAL_DISPLACEMENTS, 
        const.N_COARSE_INITIAL_VELOCITIES,
        model.n_modes * 2
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)
    
    elapsed = time.time() - start_time
    sims_per_sec = n_simulations / elapsed
    sims_per_sec_formatted = f"{sims_per_sec:,.0f}".replace(",", ".")
    print(f"-> completed in {elapsed:.3f} seconds ", f"({sims_per_sec_formatted} simulations/sec)")

    return t_max_disp, y_max_disp, t_max_vel, y_max_vel

def _select_branches(
    t_max_disp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)
    y_max_disp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, 2 * n_modes)
    sweep_direction: SweepDirection,
) -> tuple[jax.Array, jax.Array]:

    n_coarse_freq, n_coarse_amp, n_init_disp, n_init_vel, n_modes = t_max_disp.shape
    n_state      = 2 * n_modes
    n_branches   = n_init_disp * n_init_vel

    t_branches = t_max_disp.reshape(n_coarse_freq, n_coarse_amp, n_branches, n_modes) # (n_coarse_freq, n_coarse_amp, n_branches, n_modes)
    y_branches = y_max_disp.reshape(n_coarse_freq, n_coarse_amp, n_branches, n_state) # (n_coarse_freq, n_coarse_amp, n_branches, n_modes * 2)

    is_fwd = sweep_direction == const.SweepDirection.FORWARD
    idx_fwd = jnp.arange(n_coarse_freq, dtype=int) # (n_coarse_freq,)
    idx_bwd = jnp.arange(n_coarse_freq - 1, -1, -1, dtype=int) # (n_coarse_freq,)

    def _pick_for_amplitude(t_seq, y_seq): # (n_coarse_freq, n_branches, ..)
        sweep_order = jax.lax.cond(is_fwd, lambda _: idx_fwd, lambda _: idx_bwd, operand=None)
        t_ord, y_ord = t_seq[sweep_order], y_seq[sweep_order]

        # start branch = state with min ‖y‖² at first frequency step
        idx0   = jnp.argmin(jnp.sum(y_ord[0] ** 2, axis=-1))
        y0     = y_ord[0, idx0]
        t0     = t_ord[0, idx0]

        def _step(prev_y, xy):
            t_cur, y_cur = xy
            dist  = jnp.sum((y_cur - prev_y) ** 2, axis=-1)
            idx   = jnp.argmin(dist)
            y_sel = y_cur[idx]
            t_sel = t_cur[idx]
            return y_sel, (t_sel, y_sel)

        _, (t_rest, y_rest) = jax.lax.scan(_step, y0, (t_ord[1:], y_ord[1:]))
        t_sel_ord = jnp.concatenate((t0[None, :], t_rest), axis=0)
        y_sel_ord = jnp.concatenate((y0[None, :], y_rest), axis=0)

        t_sel = jax.lax.cond(is_fwd, lambda _: t_sel_ord, lambda _: t_sel_ord[::-1], operand=None)
        y_sel = jax.lax.cond(is_fwd, lambda _: y_sel_ord, lambda _: y_sel_ord[::-1], operand=None)
        return t_sel, y_sel

    t_max_disp, y_max_disp = jax.vmap(_pick_for_amplitude, in_axes=(1, 1), out_axes=(1, 1))(t_branches, y_branches)

    return t_max_disp, y_max_disp

def _fine_sweep(
    model: AbstractModel,
    t_max_disp: jax.Array,        # (n_coarse_freq, n_coarse_amp, n_modes)  – unused here
    y_max_disp: jax.Array,        # (n_coarse_freq, n_coarse_amp, 2*n_modes)
    driving_frequencies: jax.Array,   # (n_freq,)
    driving_amplitudes: jax.Array,    # (n_amp,)
    solver: AbstractSolver,
) -> tuple[jax.Array, jax.Array]:
    
    n_simulations = driving_frequencies.shape[0] * driving_amplitudes.shape[0]
    n_simulations_formatted = f"{n_simulations:,}".replace(",", ".")
    print("\nFine sweep:")
    print(f"-> running {n_simulations_formatted} simulations in parallel...")
    start_time = time.time()

    import matplotlib.pyplot as plt

    n_modes  = model.n_modes
    n_state  = 2 * n_modes
    n_freq   = driving_frequencies.shape[0]
    n_amp    = driving_amplitudes.shape[0]

    # coarse ➜ fine polynomial fit of initial conditions
    deg = 3  # polynomial degree in each variable
    # define coarse grid coordinates
    f_coarse = jnp.linspace(jnp.min(driving_frequencies),
                            jnp.max(driving_frequencies),
                            y_max_disp.shape[0])
    a_coarse = jnp.linspace(jnp.min(driving_amplitudes),
                            jnp.max(driving_amplitudes),
                            y_max_disp.shape[1])
    F_c, A_c = jnp.meshgrid(f_coarse, a_coarse, indexing="ij")
    # flatten coarse data
    F_flat = F_c.ravel()
    A_flat = A_c.ravel()
    Y_flat = y_max_disp.reshape(-1, n_state)  # (n_coarse_freq * n_coarse_amp, n_state)

    # build design matrix using all monomials f^i * a^j with i+j <= deg
    powers = [(i, j) for i in range(deg+1) for j in range(deg+1-i)]
    X = jnp.stack([ (F_flat**i) * (A_flat**j) for (i, j) in powers ], axis=1)
    # solve least-squares for each state dimension
    coeffs, *_ = jnp.linalg.lstsq(X, Y_flat, rcond=None)  # (n_monomials, n_state)

    # --- plot fit vs. data for the first state dof (e.g. mode‐0 displacement) ---
    # evaluate fit back on the coarse grid
    Y_fit_flat = X @ coeffs                        # (n_coarse_freq * n_coarse_amp, n_state)
    Z_data = Y_flat[:, 0].reshape(F_c.shape)       # true y for mode0
    Z_fit  = Y_fit_flat[:, 0].reshape(F_c.shape)   # fitted y

    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax1.plot_surface(F_c, A_c, Z_data, cmap='viridis', alpha=0.7)
    ax1.set_title('Coarse data (mode 0)')
    ax1.set_xlabel('Frequency'); ax1.set_ylabel('Amplitude'); ax1.set_zlabel('y')

    ax2 = fig.add_subplot(1,2,2, projection='3d')
    ax2.plot_surface(F_c, A_c, Z_fit, cmap='plasma', alpha=0.7)
    ax2.set_title('Polynomial fit (mode 0)')
    ax2.set_xlabel('Frequency'); ax2.set_ylabel('Amplitude'); ax2.set_zlabel('y')

    plt.tight_layout()
    plt.show()
    # ---------------------------------------------------------------------------

    # now evaluate polynomial on the fine grid for use as initial conds
    F_f, A_f = jnp.meshgrid(driving_frequencies, driving_amplitudes, indexing="ij")
    Ff_flat = F_f.ravel()
    Af_flat = A_f.ravel()
    X_fine = jnp.stack([ (Ff_flat**i) * (Af_flat**j) for (i, j) in powers ], axis=1)
    y_init_flat = X_fine @ coeffs  # (n_freq*n_amp, n_state)
    y_init_fine = y_init_flat.reshape(n_freq, n_amp, n_state)

    # build simulation grid
    freq_mesh, amp_mesh = jnp.meshgrid(
        driving_frequencies, driving_amplitudes, indexing="ij"
    )
    freq_flat = freq_mesh.ravel()
    amp_flat  = amp_mesh.ravel()
    y0_flat   = y_init_fine.reshape(-1, n_state)

    @jax.jit
    def _solve(freq, amp, y0):
        return solver(model, freq, amp, y0, const.ResponseType.FrequencyResponse)

    ts, ys = jax.vmap(_solve)(freq_flat, amp_flat, y0_flat)

    ss_disp = ys[..., :n_modes]
    ss_vel = ys[..., n_modes:]

    max_disp = jnp.max(jnp.abs(ss_disp), axis=1)  
    max_vel  = jnp.max(jnp.abs(ss_vel),  axis=1) 

    elapsed = time.time() - start_time
    sims_per_sec = n_simulations / elapsed
    sims_per_sec_formatted = f"{sims_per_sec:,.0f}".replace(",", ".")
    print(f"-> completed in {elapsed:.3f} seconds ", f"({sims_per_sec_formatted} simulations/sec)")

    return max_disp, max_vel

def frequency_sweep(
    model: AbstractModel,
    sweep_direction: SweepDirection,
    driving_frequencies: jax.Array, # Shape: (n_driving_frequencies,)
    driving_amplitudes: jax.Array, # Shape: (n_driving_amplitudes,)(n_driving_frequencies * n_driving_amplitudes, n_modes)
    solver: AbstractSolver,
) -> tuple:
            
    if isinstance(solver, SteadyStateSolver) and jnp.any(driving_frequencies == 0):
        raise TypeError("SteadyStateSolver is not compatible with zero driving frequency. Use StandardSolver for zero frequency cases (free vibration).")
    
    if solver.n_time_steps is None:
        '''
        ASSUMPTION: Minimum required sampling frequency for the steady state solver based on gpt prompt, should investigate
        '''
        rtol = 0.001
        max_driving_frequency = jnp.max(driving_frequencies)
        max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * max_driving_frequency
        
        one_period = 2.0 * jnp.pi / max_frequency_component
        sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component * 1.05 # ASSUMPTION: 1.05 is a safety factor to ensure the sampling frequency is above the Nyquist rate
        
        n_time_steps = int(jnp.ceil(one_period * sampling_frequency))
        solver.n_time_steps = n_time_steps

        print("\nAutomatically determined number of time steps for steady state solver:", n_time_steps)

    # We start by exploring the branches of the system
    t_max_disp, y_max_disp, t_max_vel, y_max_vel = _explore_branches(
        model=model,
        drive_freq=driving_frequencies,
        drive_amp=driving_amplitudes,
        solver=solver,
        sweep_direction=sweep_direction,
    )
    # t: # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)
    # y: # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)
    
    # Based on if it is a forward or backward sweep, we select the right branches
    t_max_disp_sel, y_max_disp_sel = _select_branches(
        t_max_disp=t_max_disp,
        y_max_disp=y_max_disp,
        sweep_direction=sweep_direction,
    ) 
    # t: (n_coarse_freq, n_coarse_amp, n_modes)
    # y: (n_coarse_freq, n_coarse_amp, n_modes * 2)
    
    # # extract the coarse‐grid steady‐state amplitudes from the selected branches
    ss_disp_amp = jnp.abs(y_max_disp_sel[..., :model.n_modes])   # (n_coarse_freq, n_coarse_amp, n_modes)
    coarse_drive_freq = jnp.linspace(jnp.min(driving_frequencies), jnp.max(driving_frequencies), ss_disp_amp.shape[0]) # (n_coarse_freq,)
    coarse_drive_amp = jnp.linspace(jnp.min(driving_amplitudes), jnp.max(driving_amplitudes), ss_disp_amp.shape[1]) # (n_coarse_amp,)

    # import matplotlib.pyplot as plt

    # # number of coarse amplitudes
    # n_a = coarse_drive_amp.shape[0]

    # # generate a reversed gray‐scale palette from light to dark
    # gray_colors = plt.cm.gray(np.linspace(0.1, 0.9, n_a))[::-1]

    # fig, ax = plt.subplots()
    # for ia, amp in enumerate(coarse_drive_amp):
    #     color = gray_colors[ia]
    #     # steady‐state displacement amplitude for mode 0 at this amplitude
    #     disp_curve = ss_disp_amp[:, ia, 0]
    #     ax.plot(coarse_drive_freq, disp_curve, color=color, label=f"A={amp:.2f}")

    # ax.set_xlabel("Driving frequency")
    # ax.set_ylabel("Steady‐state displacement amplitude (mode 0)")
    # plt.tight_layout()
    # plt.show()
    

    ss_disp_amp, ss_vel_amp = _fine_sweep(
        model=model,
        t_max_disp=t_max_disp_sel,
        y_max_disp=y_max_disp_sel,
        driving_frequencies=driving_frequencies,
        driving_amplitudes=driving_amplitudes,
        solver=solver,
    )

    # ASSUMPTION: Mode superposition validity for nonlinear systems
    tot_ss_disp_amp = jnp.sum(ss_disp_amp, axis=1)
    tot_ss_vel_amp = jnp.sum(ss_vel_amp, axis=1)

    frequency_sweep = FrequencySweepResult(
        model=model,
        sweep_direction=sweep_direction,
        driving_frequencies=driving_frequencies, # Shape: (n_driving_frequencies,)
        driving_amplitudes=driving_amplitudes, # Shape: (n_driving_amplitudes,)
        steady_state_displacement_amplitude=ss_disp_amp, # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        steady_state_velocity_amplitude=ss_vel_amp, # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        total_steady_state_displacement_amplitude=tot_ss_disp_amp, # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        total_steady_state_velocity_amplitude=tot_ss_vel_amp, # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        solver=solver,
    )

    return frequency_sweep