from __future__ import annotations
import jax
import jax.numpy as jnp
import time

from .model.abstract_model import AbstractModel
from .solver.abstract_solver import AbstractSolver
from .solver.time_integration_solver import TimeIntegrationSolver
from . import constants as const


def time_response(
    model: AbstractModel,
    driving_frequency: jax.Array, # Shape: (1,)
    driving_amplitude: jax.Array, # Shape: (n_modes,)
    initial_displacement: jax.Array, # Shape: (n_modes,)
    initial_velocity: jax.Array, # Shape: (n_modes,)
    solver: AbstractSolver = TimeIntegrationSolver(),
    precision: const.Precision = const.Precision.DOUBLE,
    **kwargs
) -> tuple:
    
    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # Ensure inputs are jax arrays (handles Python floats, lists, numpy arrays) before dtype enforcement
    driving_frequency = jnp.asarray(driving_frequency, dtype=dtype)
    driving_amplitude = jnp.asarray(driving_amplitude, dtype=dtype)
    initial_displacement = jnp.asarray(initial_displacement, dtype=dtype)
    initial_velocity = jnp.asarray(initial_velocity, dtype=dtype)
    
    if model.n_modes != initial_displacement.size:
        raise ValueError(f"Model has {model.n_modes} modes, but initial displacement has shape {initial_displacement.shape}. It should have shape ({model.n_modes},).")
    if model.n_modes != initial_velocity.size:
        raise ValueError(f"Model has {model.n_modes} modes, but initial velocity has shape {initial_velocity.shape}. It should have shape ({model.n_modes},).")
    
    model = model.to_dtype(dtype)
    solver.model = model

    print("Time response: ", model)
    start_time = time.time()
    ts, ys = solver.time_response(driving_frequency, driving_amplitude, initial_displacement, initial_velocity, **kwargs)
    print("Time response completed in {:.2f} seconds".format(time.time() - start_time))


    xs = ys[:, :model.n_modes]  # Shape: (n_steps, n_modes)
    vs = ys[:, model.n_modes:]  # Shape: (n_steps, n_modes)

    return ts, xs, vs