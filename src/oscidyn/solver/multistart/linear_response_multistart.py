import jax.numpy as jnp

from .abstract_multistart import AbstractMultistart

class LinearResponseMultistart(AbstractMultistart):
    def __init__(self, init_cond_shape: tuple = (11, 11), linear_response_factor: float = 1.5):
        super().__init__()
        self.init_cond_shape = init_cond_shape
        self.linear_response_factor = linear_response_factor

    def generate_simulation_grid(self, model, f_omega, f_amp):
        max_abs_displacements = (f_amp * model.Q / model.omega_0**2) * self.linear_response_factor

        if self.init_cond_shape[0] > 1:
            x0_grid = jnp.linspace(
                -1.0, 1.0, self.init_cond_shape[0]
            ) * max_abs_displacements[:, None]
        else:
            x0_grid = jnp.zeros((f_amp.shape[0], 1))

        if self.init_cond_shape[1] > 1:
            v0_grid = jnp.linspace(
                -1.0, 1.0, self.init_cond_shape[1]
            ) * max_abs_displacements[:, None]
        else:
            v0_grid = jnp.zeros((f_amp.shape[0], 1))

        f_omega_mesh, f_amp_idx_mesh, x0_idx_mesh, v0_idx_mesh = jnp.meshgrid(
            f_omega,
            jnp.arange(f_amp.shape[0]),
            jnp.arange(self.init_cond_shape[0]),
            jnp.arange(self.init_cond_shape[1]),
            indexing="ij",
        )

        f_amp_mesh = f_amp[f_amp_idx_mesh]
        x0_mesh = x0_grid[f_amp_idx_mesh, x0_idx_mesh]
        v0_mesh = v0_grid[f_amp_idx_mesh, v0_idx_mesh]

        return (f_omega_mesh, f_amp_mesh, x0_mesh, v0_mesh)