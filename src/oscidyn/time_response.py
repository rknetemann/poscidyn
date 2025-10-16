from __future__ import annotations
import jax
import jax.numpy as jnp
import time

from .model.abstract_model import AbstractModel
from .solver.abstract_solver import AbstractSolver
from .solver.steady_state_window_solver import SteadyStateSolver
from . import constants as const


def time_response(
    model: AbstractModel,
    driving_frequency: jax.Array, # Shape: (1,)
    driving_amplitude: jax.Array, # Shape: (n_modes,)
    initial_displacement: jax.Array, # Shape: (n_modes,)
    initial_velocity: jax.Array, # Shape: (n_modes,)
    solver: AbstractSolver,
    precision: const.Precision = const.Precision.DOUBLE,
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
    
    if solver.n_time_steps is None:
        '''
        ASSUMPTION: Minimum required sampling frequency for the steady state solver based on gpt prompt, should investigate
        '''
        rtol = 0.001
        max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * driving_frequency
        
        one_period = 2.0 * jnp.pi / max_frequency_component
        sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component * 1.05 # ASSUMPTION: 1.05 is a safety factor to ensure the sampling frequency is above the Nyquist rate
        
        n_time_steps = int(jnp.ceil(one_period * sampling_frequency))
        solver.n_time_steps = n_time_steps

        print("\nAutomatically determined number of time steps for steady state solver:", n_time_steps)

    model = model.to_dtype(dtype)
    solver.model = model

    print("Time response: ", model)
    start_time = time.time()
    ts, ys = solver.time_response(driving_frequency, driving_amplitude, initial_displacement, initial_velocity)
    print("Time response completed in {:.2f} seconds".format(time.time() - start_time))


    xs = ys[:, :model.n_modes]  # Shape: (n_steps, n_modes)
    vs = ys[:, model.n_modes:]  # Shape: (n_steps, n_modes)

    return ts, xs, vs