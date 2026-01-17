import jax.numpy as jnp

from .abstract_multistart import AbstractMultistart

class LinearResponseMultistart(AbstractMultistart):
    def __init__(self, init_cond_shape: tuple = (3, 3), linear_response_factor: float = 1.0):
        super().__init__()
        self.init_cond_shape = init_cond_shape
        self.linear_response_factor = linear_response_factor

    def generate_simulation_grid(self, model, f_omegas, f_amps):
        max_force = jnp.max(f_amps, axis=None)
        max_displacement_per_mode = (max_force * model.Q / model.omega_0**2) * self.linear_response_factor

        n_f_omegas = f_omegas.shape[0]
        n_f_amps = f_amps.shape[0]
        n_x0s = self.init_cond_shape[0]
        n_v0s = self.init_cond_shape[1]
        n_dof = model.n_dof

        f_omegas_grid = jnp.tile(f_omegas[:, None], (1, n_dof))
        f_amps_grid = f_amps

        if n_x0s > 1:
            x0s_grid = jnp.outer(jnp.linspace(
                -1.0, 1.0, n_x0s
            ), max_displacement_per_mode)

        else:
            x0s_grid = jnp.zeros((1, n_dof))

        if n_v0s > 1:
            v0s_grid = jnp.outer(jnp.linspace(
                -1.0, 1.0, n_v0s
            ), max_displacement_per_mode)
        else:
            v0s_grid = jnp.zeros((1, n_dof))

        f_omega_idx_mesh, f_amp_idx_mesh, x0_idx_mesh, v0_idx_mesh, mode_idx_mesh = jnp.meshgrid(
            jnp.arange(n_f_omegas),
            jnp.arange(n_f_amps),
            jnp.arange(n_x0s),
            jnp.arange(n_v0s),
            jnp.arange(n_dof),
            indexing="ij",
        )

        f_omegas_mesh = f_omegas_grid[f_omega_idx_mesh, mode_idx_mesh]
        f_amps_mesh = f_amps_grid[f_amp_idx_mesh, mode_idx_mesh]
        x0_mesh = x0s_grid[x0_idx_mesh, mode_idx_mesh]
        v0_mesh = v0s_grid[v0_idx_mesh, mode_idx_mesh]

        n_combinations = n_f_omegas * n_f_amps * n_x0s * n_v0s
        f_omegas_mesh = f_omegas_mesh.reshape(n_combinations, n_dof)
        f_amps_mesh = f_amps_mesh.reshape(n_combinations, n_dof)
        x0_mesh = x0_mesh.reshape(n_combinations, n_dof)
        v0_mesh = v0_mesh.reshape(n_combinations, n_dof)

        shape = (n_f_omegas, n_f_amps, n_x0s, n_v0s, n_dof)  # Update shape to reflect the full grid dimensions

        return (f_omegas_mesh, f_amps_mesh, x0_mesh, v0_mesh, shape)
    
    def to_dtype(self, dtype: jnp.dtype):
        return LinearResponseMultistart(
            init_cond_shape=self.init_cond_shape,
            linear_response_factor=self.linear_response_factor
        )