import jax.numpy as jnp

from .abstract_multistart import AbstractMultistart

class LinearResponseMultistart(AbstractMultistart):
    def __init__(self, init_cond_shape: int = 21, linear_response_factor: float = 1.5):
        super().__init__()
        self.init_cond_shape = init_cond_shape
        self.linear_response_factor = linear_response_factor

    def generate_simulation_grid(self, model, drive_freq, drive_amp):
        self.max_abs_displacement = float((jnp.max(drive_amp) * jnp.abs(model.Q)).item()) * self.linear_response_factor

        if self.init_cond_shape[0] > 1:
            init_disp_grid = jnp.linspace(
                -self.max_abs_displacement, self.max_abs_displacement, self.init_cond_shape[0]
            ) # (N_COARSE_INITIAL_DISPLACEMENTS,)
        else:
            init_disp_grid = jnp.array([0.0])

        if self.init_cond_shape[1] > 1:
            init_vel_grid = jnp.linspace(
                -self.max_abs_displacement, self.max_abs_displacement, self.init_cond_shape[1]
            ) # (N_COARSE_INITIAL_VELOCITIES,)
        else:
            init_vel_grid = jnp.array([0.0])  # Forcing initial velocity to be zero for now

        drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = jnp.meshgrid(
            drive_freq, drive_amp, init_disp_grid, init_vel_grid, indexing="ij"
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES)

        return (drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh)