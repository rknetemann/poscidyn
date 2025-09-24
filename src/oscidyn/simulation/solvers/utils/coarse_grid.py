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
        linear_response_amplitude = coarse_drive_amp[:, None] / denom[None, :]

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

        # 3) Flatten if you still want 1D lists of scenarios
        coarse_drive_freq_flat = coarse_drive_freq_mesh.ravel()
        coarse_drive_amp_flat  = coarse_drive_amp_mesh.ravel()
        coarse_init_disp_flat  = coarse_init_disp_mesh.ravel()
        coarse_init_vel_flat   = coarse_init_vel_mesh.ravel()

        return (coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat)