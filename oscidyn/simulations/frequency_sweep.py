# nonlinear_dynamics.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple
import numpy as np

from oscidyn.models import AbstractModel
from oscidyn.solver import AbstractSolver,SteadyStateSolver
from oscidyn.constants import SweepDirection
from oscidyn.results import FrequencySweepResult
import time
import oscidyn.constants as const

# TO DO: Improve steady state amplitude calculation
# ASSUMPTION: The steady state is already reached in the time response
@jax.jit
def _get_steady_state_amplitudes(
    drive_freq_flat: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES,)
    time_flat: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps)
    ss_disp_flat: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)
    ss_vel_flat: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)
):
    _n_sim, n_time_steps, _n_modes = ss_disp_flat.shape

    dt = time_flat[:, 1] - time_flat[:, 0] # (n_sim,)
    
    samp_per_per = jnp.round((2.0 * jnp.pi / drive_freq_flat) / dt).astype(int)

    time_idx   = jnp.arange(n_time_steps)[None, :]
    start_idx  = n_time_steps - const.N_PERIODS_TO_RETAIN * samp_per_per[:, None]
    in_window  = time_idx >= start_idx # bool (n_sim, n_time_steps)

    in_window3 = in_window[..., None]

    ss_disp_amp_flat = jnp.max(jnp.where(in_window3, jnp.abs(ss_disp_flat), 0.0), axis=1)
    ss_vel_amp_flat  = jnp.max(jnp.where(in_window3, jnp.abs(ss_vel_flat),  0.0), axis=1)

    return ss_disp_amp_flat, ss_vel_amp_flat
    
# ASSUMPTION: Initial velocity does not have to be tested for finding branches
def _explore_branches(
        model: AbstractModel,
        drive_freq: jax.Array, # (n_driving_frequencies,)
        drive_amp:  jax.Array, # (n_driving_amplitudes,)
        solver: AbstractSolver,
        sweep_direction: SweepDirection,
    ) -> Tuple[jax.Array, jax.Array]:
        
        n_simulations = const.N_COARSE_DRIVING_FREQUENCIES * const.N_COARSE_DRIVING_AMPLITUDES \
        * const.N_COARSE_INITIAL_DISPLACEMENTS 
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

        max_displacement = 20.0 # TO DO: Determine the max amplitude based on the model or a fixed value
        coarse_init_disp = jnp.linspace(
            0.01, max_displacement, const.N_COARSE_INITIAL_DISPLACEMENTS
        ) # (N_COARSE_INITIAL_DISPLACEMENTS,)

        coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh = jnp.meshgrid(
            coarse_drive_freq, coarse_drive_amp, coarse_init_disp, indexing="ij"
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS)

        coarse_drive_freq_flat = coarse_drive_freq_mesh.ravel()
        coarse_drive_amp_flat = coarse_drive_amp_mesh.ravel()
        coarse_init_disp_flat = coarse_init_disp_mesh.ravel()
        # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS)
        
        @jax.jit
        def solve_case(drive_freq, drive_amp, init_disp):
            init_disp = jnp.full((model.n_modes,), init_disp)
            init_vel = jnp.zeros((model.n_modes,))
            init_cond = jnp.concatenate([init_disp, init_vel])

            return solver(model, drive_freq, drive_amp, init_cond, const.ResponseType.FrequencyResponse)    
        
        sol = jax.vmap(solve_case)( coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat)

        if isinstance(solver, SteadyStateSolver):
            ts, ys = sol

            disp = ys[..., :model.n_modes] # (n_sim, n_windows, n_steps, n_modes)
            vel  = ys[..., model.n_modes:] # (n_sim, n_windows, n_steps, n_modes)

            sample_mask = jnp.any(jnp.abs(ys) > 0, axis=-1) # (n_sim, n_windows, n_steps)
            sample_mask = sample_mask[..., None] # broadcast over modes

            ss_disp_amp_mdir = jnp.max(jnp.where(sample_mask, jnp.abs(disp), 0.0), axis=(1, 2)) # (n_sim, n_modes)
            ss_vel_amp_mdir  = jnp.max(jnp.where(sample_mask, jnp.abs(vel),  0.0), axis=(1, 2)) # (n_sim, n_modes)
        else:
            ts, ys = sol

            time_flat = ts
            steady_state_displacements_flat = ys[..., :model.n_modes] # (n_sim, n_windows, n_steps, n_modes)
            steady_state_velocities_flat = ys[..., model.n_modes:] # (n_sim, n_windows, n_steps, n_modes)

            ss_disp_amp_mdir , ss_vel_amp_mdir  = _get_steady_state_amplitudes(
                coarse_drive_freq_flat,
                time_flat,
                steady_state_displacements_flat,
                steady_state_velocities_flat
            ) # (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        ss_disp_amp_mdir = ss_disp_amp_mdir.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        ss_disp_vel_mdir = ss_vel_amp_mdir.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS,
            model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        elapsed = time.time() - start_time
        sims_per_sec = n_simulations / elapsed
        sims_per_sec_formatted = f"{sims_per_sec:,.0f}".replace(",", ".")
        print(f"-> completed in {elapsed:.3f} seconds ", f"({sims_per_sec_formatted} simulations/sec)")

        return ss_disp_amp_mdir, ss_disp_vel_mdir
    
# TO DO: Improve branch selection logic
def _select_branches(
    model: AbstractModel,
    drive_freq: jax.Array, # (n_driving_frequencies,)
    coarse_drive_freq: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES,)
    ss_disp_amp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)
    ss_vel_amp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)
    sweep_direction: SweepDirection,
):

    norm = jnp.linalg.norm(ss_disp_amp, axis=-1)

    if sweep_direction is SweepDirection.FORWARD:     # large-amplitude branch
        best_idx = jnp.argmax(norm, axis=2)           # (freq, amp)
    else:                                             # small-amplitude branch
        best_idx = jnp.argmin(norm, axis=2)

    def _gather(arr):
        # arr shape (freq, amp, init, mode)
        idx_exp = best_idx[..., None, None]           # expand for take_along
        out     = jnp.take_along_axis(arr, idx_exp, axis=2)
        return out.squeeze(axis=2)                    # → (freq, amp, mode)

    disp_branch = _gather(ss_disp_amp)                  # (freq, amp, mode)
    vel_branch  = _gather(ss_vel_amp)

    def _interp_over_freq(coarse_array: jax.Array) -> jax.Array:
        # array_coarse has shape (n_freq_c, n_amp_c, mode)
        def _one_mode(a_mode):
            return jnp.stack([
                jnp.interp(drive_freq, coarse_drive_freq, a_mode[:, j])
                for j in range(const.N_COARSE_DRIVING_AMPLITUDES)
            ], axis=1) # (fine_freq, amp_c)
        return jax.vmap(_one_mode, in_axes=2, out_axes=2)(coarse_array)

    disp_fine = _interp_over_freq(disp_branch) # (fine_freq, amp_c, mode)
    vel_fine  = _interp_over_freq(vel_branch) # (fine_freq, amp_c, mode)

    return disp_fine, vel_fine

def _fine_sweep(
    model: AbstractModel,
    ss_disp_amp: jax.Array, 
    ss_vel_amp: jax.Array,
    driving_frequencies: jax.Array,          # (n_f,)
    driving_amplitudes:  jax.Array,          # (n_A_fine,)
    solver: AbstractSolver,
) -> tuple[jax.Array, jax.Array]:

    # ASSUMPTION: The measured steady-state amplitude is exactly at the peak, and thus velocity is zero.
    batch_shape = ss_disp_amp.shape[:-1]
    initial_conditions = jnp.concatenate([
        ss_disp_amp.reshape(*batch_shape, model.n_modes),
        jnp.zeros((*batch_shape, model.n_modes))
    ], axis=-1)  # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, 2 * n_modes)
    
    n_f           = driving_frequencies.size
    n_A_fine      = driving_amplitudes.size
    n_A_coarse    = initial_conditions.shape[1]
    n_params      = initial_conditions.shape[2]        # 2 * n_modes
    n_modes       = model.n_modes
    
    n_simulations = driving_frequencies.size * driving_amplitudes.size
    n_simulations_formatted = f"{n_simulations:,}".replace(",", ".")
    print("\nFine sweep:")
    print(f"-> running {n_simulations_formatted} simulations in parallel...")


    # ------------------------------------------------------------------
    # 1.  If necessary, interpolate ICs from the coarse → fine amplitude grid
    # ------------------------------------------------------------------
    if n_A_coarse != n_A_fine:
        coarse_amps = jnp.linspace(
            driving_amplitudes[0], driving_amplitudes[-1], n_A_coarse
        )                                             # (n_A_coarse,)

        # ----  vmap over frequency and over the 2*n_modes parameters ----
        def _interp_row(ic_row):                      # ic_row (n_A_coarse, n_params)
            ic_T        = ic_row.T                   # (n_params, n_A_coarse)
            interp_T    = jax.vmap(
                lambda param_vals:
                    jnp.interp(driving_amplitudes, coarse_amps, param_vals)
            )(ic_T)                                  # (n_params, n_A_fine)
            return interp_T.T                        # → (n_A_fine, n_params)

        initial_conditions = jax.vmap(_interp_row)(initial_conditions)
        # shape now (n_f, n_A_fine, 2*n_modes)

    freq_mesh, amp_mesh = jnp.meshgrid(driving_frequencies, driving_amplitudes, indexing="ij") # both (n_f, n_A_fine)

    freq_flat = freq_mesh.ravel() # (n_f * n_A_fine,)
    amp_flat  = amp_mesh .ravel()
    ic_flat   = initial_conditions.reshape(-1, n_params) # (n_f * n_A_fine, 2*n_modes)


    @jax.jit
    def solve_case(driving_frequency, driving_amplitude, initial_condition):
        return solver(model, driving_frequency, driving_amplitude, initial_condition, const.ResponseType.FrequencyResponse)
    
    start_time = time.time()
    if isinstance(solver, SteadyStateSolver):
        ts, ys = jax.vmap(solve_case, in_axes=(0, 0, 0))(freq_flat, amp_flat, ic_flat)
        
        disp = ys[..., :n_modes]         # (n_sim, n_windows, n_steps, n_modes)
        vel  = ys[..., n_modes:]         # (n_sim, n_windows, n_steps, n_modes)

        # Build a mask that is True on samples that actually contain data (all‑zero windows produced by SteadyStateSolver are ignored)
        sample_mask = jnp.any(jnp.abs(ys) > 0, axis=-1) # (n_sim, n_windows, n_steps)
        sample_mask = sample_mask[..., None]

        # Peak amplitudes = max over (windows, time) of *valid* samples
        ss_disp_peak = jnp.max(jnp.where(sample_mask, jnp.abs(disp), 0.0), axis=(1, 2)) # (n_sim, n_modes)
        ss_vel_peak  = jnp.max(jnp.where(sample_mask, jnp.abs(vel),  0.0), axis=(1, 2)) # (n_sim, n_modes)
    else:
        ts, ys = jax.vmap(solve_case, in_axes=(0, 0, 0))(freq_flat, amp_flat, ic_flat)
        
        ss_disp = ys[..., :n_modes]  # (n_sim, n_windows, n_steps, n_modes)
        ss_vel = ys[..., n_modes:]     # (n_sim, n_windows, n_steps, n_modes)

        ss_disp_peak, ss_vel_peak = _get_steady_state_amplitudes(freq_flat, ts, ss_disp, ss_vel) # (n_sim, n_modes)
        
    elapsed = time.time() - start_time
    sims_per_sec = n_simulations / elapsed
    sims_per_sec_formatted = f"{sims_per_sec:,.0f}".replace(",", ".")
    print(f"-> completed in {elapsed:.3f} seconds "f"({sims_per_sec_formatted} simulations/sec)")       

    return ss_disp_peak, ss_vel_peak

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

    ss_disp_amp_mdir, ss_vel_amp_mdir = _explore_branches(
        model=model,
        drive_freq=driving_frequencies,
        drive_amp=driving_amplitudes,
        solver=solver,
        sweep_direction=sweep_direction,
    ) # (n_driving_frequencies, n_driving_amplitudes, 2 * n_modes)
    
    # Generate coarse driving frequencies array for branch selection
    coarse_drive_freq = jnp.linspace(
        jnp.min(driving_frequencies), 
        jnp.max(driving_frequencies), 
        ss_disp_amp_mdir.shape[0]
    )
    
    ss_disp_amp, ss_vel_amp = _select_branches(
        model=model,
        drive_freq=driving_frequencies,
        coarse_drive_freq=coarse_drive_freq,
        ss_disp_amp=ss_disp_amp_mdir,
        ss_vel_amp=ss_vel_amp_mdir, 
        sweep_direction=sweep_direction,
    ) # (n_driving_frequencies, n_driving_amplitudes, n_modes)
    
    ss_disp_amp, ss_vel_amp = _fine_sweep(
        model=model,
        ss_disp_amp=ss_disp_amp,
        ss_vel_amp=ss_vel_amp,
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