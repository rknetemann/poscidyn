# nonlinear_dynamics.py
from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple
import numpy as np
import time

from ..models.abstract_model import AbstractModel
from ..solvers.abstract_solver import AbstractSolver
from ..solvers.steady_state_window_solver import SteadyStateSolver
from .. import constants as const

def time_response(
    model: AbstractModel,
    driving_frequency: jax.Array, # Shape: (1,)
    driving_amplitude: jax.Array, # Shape: (n_modes,)
    initial_displacement: jax.Array, # Shape: (n_modes,)
    initial_velocity: jax.Array, # Shape: (n_modes,)
    solver: AbstractSolver,
) -> tuple:
    
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

    initial_condition = jnp.concatenate([initial_displacement, initial_velocity])

    ts, ys = solver(
        model=model,
        driving_frequency=driving_frequency,
        driving_amplitude=driving_amplitude,
        initial_condition=initial_condition,
        response=const.ResponseType.TimeResponse
    )

    if isinstance(solver, SteadyStateSolver):
        time = ts.flatten()

        # Remove windows that contain only zeros, artifacts of the parallel solver
        nonzero_mask = jnp.any(jnp.abs(ys) > 0, axis=(1, 2)) # (n_windows,)
        ts_nonzero = ts[nonzero_mask] # (n_windows_nonzero, n_steps)
        ys_nonzero = ys[nonzero_mask] # (n_windows_nonzero, n_steps, 2*n_modes)

        time = ts_nonzero.flatten() # (n_windows_nonzero * n_steps,)
        displacements = ys_nonzero[:, :, :model.n_modes].reshape(-1, model.n_modes)
        velocities    = ys_nonzero[:, :, model.n_modes:].reshape(-1, model.n_modes)
    else:
        # For FixedTimeSolver, ts and ys are already in the correct shape
        time = ts
        displacements = ys[:, :model.n_modes]  # Shape: (n_steps, n_modes)
        velocities = ys[:, model.n_modes:]  # Shape: (n_steps, n_modes)

    return time, displacements, velocities