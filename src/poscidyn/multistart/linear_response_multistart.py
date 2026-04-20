import numpy as np
import jax
import jax.numpy as jnp

from .abstract_multistart import AbstractMultistart

class LinearResponseMultistart(AbstractMultistart):
    def __init__(
        self,
        n_init_cond: int = 16,
        linear_response_factor: float = 1.0,
        random_seed: int | None = 0,
    ):
        super().__init__()
        n_init_cond = int(n_init_cond)
        if n_init_cond < 1:
            raise ValueError("n_init_cond must be >= 1.")

        self.n_init_cond = n_init_cond
        self.linear_response_factor = linear_response_factor
        self.random_seed = random_seed

    def generate_simulation_grid(self, model, f_omegas, f_amps):
        max_force = jnp.max(f_amps, axis=None)
        max_displacement_per_mode = (max_force * model.Q / model.omega_0**2) * self.linear_response_factor
        max_velocity_per_mode = (max_force * model.Q / model.omega_0) * self.linear_response_factor

        n_f_omegas = f_omegas.shape[0]
        n_f_amps = f_amps.shape[0]
        n_init_cond = self.n_init_cond
        n_modes = model.n_modes

        f_omegas_grid = jnp.tile(f_omegas[:, None], (1, n_dof))
        f_amps_grid = f_amps

        if self.random_seed is None:
            random_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        else:
            random_seed = int(self.random_seed)

        key = jax.random.PRNGKey(random_seed)
        key_x0, key_v0 = jax.random.split(key)
        unit_x0s_grid = jax.random.uniform(
            key_x0,
            shape=(n_init_cond, n_modes),
            minval=-1.0,
            maxval=1.0,
            dtype=f_amps.dtype,
        )
        unit_v0s_grid = jax.random.uniform(
            key_v0,
            shape=(n_init_cond, n_modes),
            minval=-1.0,
            maxval=1.0,
            dtype=f_amps.dtype,
        )

        x0s_grid = unit_x0s_grid * max_displacement_per_mode[None, :]
        v0s_grid = unit_v0s_grid * max_velocity_per_mode[None, :]

        shape = (n_f_omegas, n_f_amps, n_init_cond, 1, n_modes)
        f_omegas_mesh = jnp.broadcast_to(f_omegas_grid[:, None, None, None, :], shape)
        f_amps_mesh = jnp.broadcast_to(f_amps_grid[None, :, None, None, :], shape)
        x0_mesh = jnp.broadcast_to(x0s_grid[None, None, :, None, :], shape)
        v0_mesh = jnp.broadcast_to(v0s_grid[None, None, :, None, :], shape)

        n_combinations = n_f_omegas * n_f_amps * n_init_cond
        f_omegas_mesh = f_omegas_mesh.reshape(n_combinations, n_modes)
        f_amps_mesh = f_amps_mesh.reshape(n_combinations, n_modes)
        x0_mesh = x0_mesh.reshape(n_combinations, n_modes)
        v0_mesh = v0_mesh.reshape(n_combinations, n_modes)

        return (f_omegas_mesh, f_amps_mesh, x0_mesh, v0_mesh, shape)
    
    def to_dtype(self, dtype: jnp.dtype):
        return LinearResponseMultistart(
            n_init_cond=self.n_init_cond,
            linear_response_factor=self.linear_response_factor,
            random_seed=self.random_seed,
        )
