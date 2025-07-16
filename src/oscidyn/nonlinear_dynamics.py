# nonlinear_dynamics.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple

from oscidyn.models import AbstractModel
from oscidyn.solver import AbstractSolver
from oscidyn.constants import SweepDirection
from oscidyn.results import FrequencySweepResult
import numpy as np
import oscidyn.constants as const

# TO DO: Improve steady state amplitude calculation
# ASSUMPTION: The steady state is already reached in the time response
@jax.jit
def _get_steady_state_amplitudes(
    driving_frequencies_flat: jnp.ndarray, # Shape: (N_COARSE_DRIVING_FREQUENCIES,)
    time_flat: jnp.ndarray, # Shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps)
    steady_state_displacements_flat: jnp.ndarray, # Shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)
    steady_state_velocities_flat: jnp.ndarray, # Shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)
):
    """
    Compute steady-state displacement and velocity amplitudes.

    For each simulation we keep only the samples that lie in the **last three
    periods** of its drive frequency and then take `max(|·|)` over that window.

    Args
    ----
    driving_frequencies: Array of driving frequencies
        (shape: [N_COARSE_DRIVING_FREQUENCIES])
    time: Array of time values
        (shape: [N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps])
    steady_state_displacements_flat: Array of steady state displacements
        (shape: [N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes])
    steady_state_velocities_flat: Array of steady state velocities
        (shape: [N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes])


    Returns
    -------
    steady_state_displacement_amplitude_flat, steady_state_velocity_amplitude_flat : jnp.ndarray
        Max absolute displacement / velocity in the last three periods for
        every simulation.  Shape (n_sim, n_modes).
    """
    n_sim, n_time_steps, n_modes = steady_state_displacements_flat.shape

    # 1.  Per-simulation dt  (works even if dt is *slightly* different between runs)
    dt = time_flat[:, 1] - time_flat[:, 0]                     # (n_sim,)

    # 2.  Samples per period  (integer, rounded)
    samp_per_per = jnp.round((2.0 * jnp.pi / driving_frequencies_flat) / dt).astype(int)

    # 3.  For every sim build a mask that is True only on the last 3 periods
    #     time_idx shape (1, n_time_steps)  → broadcast to (n_sim, n_time_steps)
    time_idx   = jnp.arange(n_time_steps)[None, :]
    start_idx  = n_time_steps - const.N_PERIODS_TO_RETAIN * samp_per_per[:, None]
    in_window  = time_idx >= start_idx               # bool (n_sim, n_time_steps)

    # 4.  Expand mask to (n_sim, n_time_steps, 1) so it broadcasts over modes
    in_window3 = in_window[..., None]

    # 5.  Apply mask, take max(|·|) over time axis
    steady_state_displacement_amplitude_flat = jnp.max(jnp.where(in_window3, jnp.abs(steady_state_displacements_flat), 0.0), axis=1)
    steady_state_velocity_amplitude_flat  = jnp.max(jnp.where(in_window3, jnp.abs(steady_state_velocities_flat),  0.0), axis=1)

    return steady_state_displacement_amplitude_flat, steady_state_velocity_amplitude_flat

# TO DO: Improve branch selection logic
def _select_branches(
    model: AbstractModel,
    driving_frequencies: jax.Array, # Shape: (n_driving_frequencies,)
    coarse_driving_frequencies: jax.Array, # Shape: (N_COARSE_DRIVING_FREQUENCIES,)
    steady_state_displacement_amplitudes, # Shape: (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)
    steady_state_velocity_amplitudes, # Shape: (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)
    sweep_direction: SweepDirection,
):
     
    norm = jnp.linalg.norm(steady_state_displacement_amplitudes, axis=-1)

    if sweep_direction is SweepDirection.FORWARD:     # large-amplitude branch
        best_idx = jnp.argmax(norm, axis=2)           # (freq, amp)
    else:                                             # small-amplitude branch
        best_idx = jnp.argmin(norm, axis=2)

    # gather helper
    def _gather(arr):
        # arr shape (freq, amp, init, mode)
        idx_exp = best_idx[..., None, None]           # expand for take_along
        out     = jnp.take_along_axis(arr, idx_exp, axis=2)
        return out.squeeze(axis=2)                    # → (freq, amp, mode)

    disp_branch = _gather(steady_state_displacement_amplitudes)                  # (freq, amp, mode)
    vel_branch  = _gather(steady_state_velocity_amplitudes)

    # ----------  6.  interpolate *only along frequency* to fine grid ----------
    # We keep the amplitude axis intact so the caller can plot a surface.
    def _interp_over_freq(array_coarse):
        # array_coarse has shape (n_freq_c, n_amp_c, mode)
        def _one_mode(a_mode):                       # (n_freq_c, n_amp_c)
            return jnp.stack([
                jnp.interp(driving_frequencies, coarse_driving_frequencies, a_mode[:, j])
                for j in range(const.N_COARSE_DRIVING_AMPLITUDES)
            ], axis=1)                               # (fine_freq, amp_c)
        return jax.vmap(_one_mode, in_axes=2, out_axes=2)(array_coarse)

    disp_fine = _interp_over_freq(disp_branch)       # (fine_freq, amp_c, mode)
    vel_fine  = _interp_over_freq(vel_branch)

    return disp_fine, vel_fine
    
# TO DO: Include the initial velocities in the initial guesses function
def _estimate_initial_conditions(
        model: AbstractModel,
        driving_frequencies: jax.Array, # Shape: (n_driving_frequencies,)
        driving_amplitudes:  jax.Array, # Shape: (n_driving_amplitudes,)
        solver: AbstractSolver,
        sweep_direction: SweepDirection,
    ) -> Tuple[jax.Array, jax.Array]:
        
        coarse_driving_frequencies = jnp.linspace(                
            jnp.min(driving_frequencies), jnp.max(driving_frequencies), const.N_COARSE_DRIVING_FREQUENCIES
        ) # Shape: (N_COARSE_DRIVING_FREQUENCIES,)

        coarse_driving_amplitudes = jnp.linspace(
            jnp.min(driving_amplitudes), jnp.max(driving_amplitudes), const.N_COARSE_DRIVING_AMPLITUDES
        ) # Shape: (N_COARSE_DRIVING_AMPLITUDES,)

        max_displacement = 1.0 # TO DO: Determine the max amplitude based on the model or a fixed value
        coarse_initial_displacement = jnp.linspace(
            0.01, max_displacement, const.N_COARSE_INITIAL_DISPLACEMENTS
        ) # Shape: (N_COARSE_INITIAL_DISPLACEMENTS,)

        coarse_driving_frequency_mesh, coarse_driving_amplitude_mesh, coarse_initial_displacement_mesh = jnp.meshgrid(
            coarse_driving_frequencies, coarse_driving_amplitudes, coarse_initial_displacement, indexing="ij"
        ) # Shape: (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS)

        coarse_driving_frequencies_flat = coarse_driving_frequency_mesh.ravel()
        coarse_driving_amplitudes_flat = coarse_driving_amplitude_mesh.ravel()
        coarse_initial_displacements_flat = coarse_initial_displacement_mesh.ravel() 
        # Shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS)

        def solve_case(driving_frequency, driving_amplitude, initial_displacement):
            initial_displacement = jnp.full((model.n_modes,), initial_displacement)
            initial_velocity = jnp.zeros((model.n_modes,))
            initial_condition = jnp.concatenate([initial_displacement, initial_velocity])

            return solver.solve_rhs(model, driving_frequency, driving_amplitude, initial_condition)

        time_flat, steady_state_displacements_flat, steady_state_velocities_flat = jax.vmap(solve_case)(
            coarse_driving_frequencies_flat, coarse_driving_amplitudes_flat, coarse_initial_displacements_flat
        )
        # time_flat shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps)
        # steady_state_displacements_flat shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)
        # steady_state_velocities_flat shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_time_steps, n_modes)

        steady_state_displacement_amplitudes_flat, steady_state_velocity_amplitudes_flat = _get_steady_state_amplitudes(
            coarse_driving_frequencies_flat,
            time_flat,
            steady_state_displacements_flat,
            steady_state_velocities_flat
        ) # Shape: (N_COARSE_DRIVING_FREQUENCIES * N_COARSE_DRIVING_AMPLITUDES * N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        steady_state_displacement_amplitudes = steady_state_displacement_amplitudes_flat.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            model.n_modes
        ) # Shape: (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        steady_state_velocity_amplitudes  = steady_state_velocity_amplitudes_flat .reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            model.n_modes
        ) # Shape: (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, n_modes)

        initial_displacement_amplitudes, initial_velocity_amplitudes = _select_branches(
            model=model,
            driving_frequencies=driving_frequencies,
            coarse_driving_frequencies=coarse_driving_frequencies,
            steady_state_displacement_amplitudes=steady_state_displacement_amplitudes,
            steady_state_velocity_amplitudes=steady_state_velocity_amplitudes,
            sweep_direction=sweep_direction,
        ) # Shape: (n_driving_frequencies, n_driving_amplitudes, n_modes)
      
        batch_shape = initial_displacement_amplitudes.shape[:-1]
        initial_conditions = jnp.concatenate([
            initial_displacement_amplitudes.reshape(*batch_shape, model.n_modes),
            initial_velocity_amplitudes.reshape(*batch_shape, model.n_modes)
        ], axis=-1) # Shape: (n_driving_frequencies, n_driving_amplitudes, 2 * n_modes)

        return initial_conditions

def _fine_sweep(
    model: AbstractModel,
    initial_conditions: jax.Array,           # (n_f, n_A_coarse, 2*n_modes)
    driving_frequencies: jax.Array,          # (n_f,)
    driving_amplitudes:  jax.Array,          # (n_A_fine,)
    solver: AbstractSolver,
) -> tuple[jax.Array, jax.Array]:
    """
    Run the fine‑grid sweep and return steady‑state displacement / velocity
    amplitudes flattened to (n_f * n_A_fine, n_modes).
    """
    n_f           = driving_frequencies.size
    n_A_fine      = driving_amplitudes.size
    n_A_coarse    = initial_conditions.shape[1]
    n_params      = initial_conditions.shape[2]        # 2 * n_modes
    n_modes       = model.n_modes

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

    # ------------------------------------------------------------------
    # 2.  Flatten sweep grid
    # ------------------------------------------------------------------
    freq_mesh, amp_mesh = jnp.meshgrid(
        driving_frequencies, driving_amplitudes, indexing="ij"
    )                                                 # both (n_f, n_A_fine)

    freq_flat = freq_mesh.ravel()                    # (n_f * n_A_fine,)
    amp_flat  = amp_mesh .ravel()
    ic_flat   = initial_conditions.reshape(
        -1, n_params
    )                                                # (n_f * n_A_fine, 2*n_modes)

    # ------------------------------------------------------------------
    # 3.  Solve every case in parallel
    # ------------------------------------------------------------------
    @jax.jit
    def _solve(freq, amp, ic):
        return solver.solve_rhs(model, freq, amp, ic)

    time_flat, disp_flat, vel_flat = jax.vmap(_solve, in_axes=(0, 0, 0))(
        freq_flat, amp_flat, ic_flat
    )
    #  → time_flat (n_sim, n_time_steps)
    #    disp_flat (n_sim, n_time_steps, n_modes)
    #    vel_flat  (n_sim, n_time_steps, n_modes)

    # ------------------------------------------------------------------
    # 4.  Extract steady‑state amplitudes
    # ------------------------------------------------------------------
    disp_amp_flat, vel_amp_flat = _get_steady_state_amplitudes(
        freq_flat, time_flat, disp_flat, vel_flat
    )                                                 # (n_sim, n_modes)

    return disp_amp_flat, vel_amp_flat


def frequency_sweep(
    model: AbstractModel,
    sweep_direction: SweepDirection,
    driving_frequencies: jax.Array, # Shape: (n_driving_frequencies,)
    driving_amplitudes: jax.Array, # Shape: (n_driving_amplitudes,)(n_driving_frequencies * n_driving_amplitudes, n_modes)
    solver: AbstractSolver,
) -> tuple:

    initial_conditions = _estimate_initial_conditions(
        model=model,
        driving_frequencies=driving_frequencies,
        driving_amplitudes=driving_amplitudes,
        solver=solver,
        sweep_direction=sweep_direction,
    ) # Shape: (n_driving_frequencies, n_driving_amplitudes, 2 * n_modes)

    steady_state_displacement_amplitudes, steady_state_velocity_amplitudes = _fine_sweep(
        model=model,
        initial_conditions=initial_conditions,
        driving_frequencies=driving_frequencies,
        driving_amplitudes=driving_amplitudes,
        solver=solver,
    )

    # ASSUMPTION: Mode superposition validity for nonlinear systems
    total_steady_state_displacement_amplitude = jnp.sum(steady_state_displacement_amplitudes, axis=1)
    total_steady_state_velocity_amplitude = jnp.sum(steady_state_velocity_amplitudes, axis=1)
    
    # define grid sizes for plotting
    n_f = driving_frequencies.size
    n_A = driving_amplitudes.size

    import matplotlib.pyplot as plt

    # reshape into (n_frequencies, n_amplitudes)
    disp_matrix = np.array(total_steady_state_displacement_amplitude).reshape(n_f, n_A)

    plt.figure()
    for idx, amp in enumerate(np.array(driving_amplitudes)):
        plt.plot(
            np.array(driving_frequencies),
            disp_matrix[:, idx],
            label=f"A = {amp:.3f}"
        )
    plt.xlabel("Driving Frequency")
    plt.ylabel("Total Steady‐State Displacement Amplitude")
    plt.title("Steady‐State Amplitude vs Frequency")
    plt.legend(title="Driving Amplitude")
    plt.grid(True)
    plt.show()

    
    frequency_sweep = FrequencySweepResult(
        model=model,
        sweep_direction=sweep_direction,
        driving_frequencies=driving_frequencies, # Shape: (n_driving_frequencies,)
        driving_amplitudes=driving_amplitudes, # Shape: (n_driving_amplitudes,)
        steady_state_displacement_amplitude=steady_state_displacement_amplitudes, # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        steady_state_velocity_amplitude=steady_state_velocity_amplitudes, # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        total_steady_state_displacement_amplitude=total_steady_state_displacement_amplitude, # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        total_steady_state_velocity_amplitude=total_steady_state_velocity_amplitude, # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        solver=solver,
    )

    return frequency_sweep

def time_response(
    model: AbstractModel,
    driving_frequency: jax.Array, # Shape: (1,)
    driving_amplitude: jax.Array, # Shape: (1,)
    initial_displacement: jax.Array, # Shape: (n_modes,)
    initial_velocity: jax.Array, # Shape: (n_modes,)
    solver: AbstractSolver,
) -> tuple:

    initial_condition = jnp.concatenate([initial_displacement, initial_velocity])

    time, displacements, velocities = solver.solve_rhs(
        model=model,
        driving_frequency=driving_frequency,
        driving_amplitude=driving_amplitude,
        initial_condition=initial_condition
    )

    return time, displacements, velocities