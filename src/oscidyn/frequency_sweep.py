from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import Tuple, Dict

from .model.abstract_model import AbstractModel
from .solver.abstract_solver import AbstractSolver
from .excitation.abstract_excitation import AbstractExcitation
from .multistart.abstract_multistart import AbstractMultistart
from .sweep.abstract_sweep import AbstractSweep
from .result.frequency_sweep_result import FrequencySweepResult 

from .solver.time_integration_solver import TimeIntegrationSolver
from .multistart.linear_response_multistart import LinearResponseMultistart
from .sweep.nearest_neighbour_sweep import NearestNeighbourSweep

from . import constants as const

def frequency_sweep(
    model: AbstractModel,
    excitor: AbstractExcitation,
    sweeper: AbstractSweep = NearestNeighbourSweep(),
    solver: AbstractSolver = TimeIntegrationSolver(),
    multistarter: AbstractMultistart = LinearResponseMultistart(),
    precision: const.Precision = const.Precision.DOUBLE,
) -> FrequencySweepResult:
            
    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    model = model.to_dtype(dtype)
    excitor = excitor.to_dtype(dtype)
    sweeper = sweeper.to_dtype(dtype)
    multistarter = multistarter.to_dtype(dtype)
    
    solver.model = model
    solver.multistarter = multistarter

    frequency_sweep = solver.frequency_sweep(excitor, sweeper)

    return frequency_sweep