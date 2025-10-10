from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from .model.abstract_model import AbstractModel
from .solver.abstract_solver import AbstractSolver
from .solver.fixed_time_solver import FixedTimeSolver
from .solver.steady_state_window_solver import SteadyStateSolver
from .solver.fixed_time_steady_state_solver import FixedTimeSteadyStateSolver
from .solver.single_shooting_solver import SingleShootingSolver
from .solver.multiple_shooting_solver import MultipleShootingSolver

from . import constants as const
from .utils import plotting as plt

def frequency_sweep(
    model: AbstractModel,
    sweep_direction: const.SweepDirection,
    driving_frequencies: jax.Array, # Shape: (n_driving_frequencies,)
    driving_amplitudes: jax.Array, # Shape: (n_driving_amplitudes,)(n_driving_frequencies * n_driving_amplitudes, n_modes)
    solver: AbstractSolver,
    precision: const.Precision = const.Precision.DOUBLE,
) -> Dict[str, jax.Array]:
            
    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # Ensure inputs are jax arrays (handles Python floats, lists, numpy arrays) before dtype enforcement
    driving_frequencies = jnp.asarray(driving_frequencies, dtype=dtype)
    driving_amplitudes = jnp.asarray(driving_amplitudes, dtype=dtype)
    
    model = model.to_dtype(dtype)
    solver.model = model

    return solver.frequency_sweep(driving_frequencies, driving_amplitudes, sweep_direction)
