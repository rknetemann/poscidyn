import jax
import jax.numpy as jnp

from ... import constants as const
from ...models import AbstractModel

def gen_coarse_grid(model: AbstractModel,
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
        linear_amplitude = coarse_drive_amp[:, None] / denom[None, :]
        print(linear_amplitude.shape)
        print(linear_amplitude)

        factor = const.N_COARSE_INITIAL_CONDITIONS_OFFSET_FACTOR
        init_disp_grid = jnp.linspace(-factor*linear_amplitude, factor*linear_amplitude, const.N_COARSE_INITIAL_DISPLACEMENTS, axis=-1)
        print(init_disp_grid.shape)
        print(init_disp_grid)



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

        return (coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat)