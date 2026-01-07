import jax
import jax.numpy as jnp

from ... import constants as const
from ...oscillator.abstract_oscillator import AbstractOscillator

# TO DO: Randomize initial conditions within the grid cells

def gen_coarse_grid_1(model: AbstractOscillator,
                    drive_freq: jax.Array,
                    drive_amp: jax.Array,
                    ):
    
    coarse_drive_freq = jnp.linspace(
        jnp.min(drive_freq), jnp.max(drive_freq), const.N_COARSE_DRIVING_FREQUENCIES
    ) # (N_COARSE_DRIVING_FREQUENCIES,)

    coarse_drive_amp = jnp.linspace(
        jnp.min(drive_amp), jnp.max(drive_amp), const.N_COARSE_DRIVING_AMPLITUDES
    ) # (N_COARSE_DRIVING_AMPLITUDES,)

    # For each combination of coarse driving frequency and amplitude, estimate the linear response amplitude
    denom = jnp.sqrt((model.omega_0**2 - coarse_drive_freq**2)**2 + (model.omega_0 * coarse_drive_freq / model.Q)**2)
    linear_response_amplitude = coarse_drive_amp[:, None] / denom[None, :]

    jax.debug.print("Max linear response amplitude: {}", jnp.max(linear_response_amplitude))

    factor = const.N_COARSE_INITIAL_CONDITIONS_OFFSET_FACTOR
    init_disp_grid = jnp.linspace(-factor*linear_response_amplitude, factor*linear_response_amplitude, const.N_COARSE_INITIAL_DISPLACEMENTS, axis=-1)
    init_vel_grid = jnp.linspace(-factor*linear_response_amplitude, factor*linear_response_amplitude, const.N_COARSE_INITIAL_VELOCITIES, axis=-1)

    F = const.N_COARSE_DRIVING_FREQUENCIES
    A = const.N_COARSE_DRIVING_AMPLITUDES
    ND = const.N_COARSE_INITIAL_DISPLACEMENTS
    NV = const.N_COARSE_INITIAL_VELOCITIES

    init_disp_FA = jnp.transpose(init_disp_grid, (1, 0, 2))  # (F, A, ND)
    init_vel_FA  = jnp.transpose(init_vel_grid,  (1, 0, 2))  # (F, A, NV)

    # 2) Broadcast everything to (F, A, ND, NV)
    coarse_drive_freq_mesh = jnp.broadcast_to(
        coarse_drive_freq[:, None, None, None], (F, A, ND, NV)
    )
    coarse_drive_amp_mesh = jnp.broadcast_to(
        coarse_drive_amp[None, :, None, None], (F, A, ND, NV)
    )
    coarse_init_disp_mesh = jnp.broadcast_to(
        init_disp_FA[..., None], (F, A, ND, NV)
    )
    coarse_init_vel_mesh = jnp.broadcast_to(
        init_vel_FA[:, :, None, :], (F, A, ND, NV)
    )

    return (coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh, coarse_init_vel_mesh)

def gen_grid_2(model: AbstractOscillator,
                    drive_freq: jax.Array,
                    drive_amp: jax.Array,
                    ):
    
    max_abs_displacement = float((jnp.max(drive_amp) * jnp.abs(model.Q)).item()) * const.LINEAR_RESPONSE_FACTOR

    init_disp_grid = jnp.linspace(
        0.0, max_abs_displacement, const.N_COARSE_INITIAL_DISPLACEMENTS
    ) # (N_COARSE_INITIAL_DISPLACEMENTS,)
    
    init_vel_grid = jnp.linspace(
        -max_abs_displacement, max_abs_displacement, const.N_COARSE_INITIAL_VELOCITIES
    ) # (N_COARSE_INITIAL_VELOCITIES,)

    drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = jnp.meshgrid(
        drive_freq, drive_amp, init_disp_grid, init_vel_grid, indexing="ij"
    ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES)

    return (drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh)