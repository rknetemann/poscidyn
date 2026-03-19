# Copyright (c) 2026 Raymond Knetemann
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
import jax
import jax.numpy as jnp

from .oscillator.abstract_oscillator import AbstractOscillator
from .solver.abstract_solver import AbstractSolver
from .excitation.abstract_excitation import AbstractExcitation
from .multistart.abstract_multistart import AbstractMultistart
from .response_measure.abstract_response_measure import AbstractResponseMeasure
from .sweep.abstract_sweep import AbstractSweep

from .excitation.one_tone import OneToneExcitation

from .result.frequency_sweep_result import FrequencySweep 

from .solver.time_integration_solver import TimeIntegrationSolver
from .multistart.linear_response_multistart import LinearResponseMultistart
from .sweep.nearest_neighbour_sweep import NearestNeighbourSweep
from .response_measure.demodulation import Demodulation

from . import constants as const

def frequency_sweep(
    model: AbstractOscillator,
    excitation: AbstractExcitation,
    sweeper: AbstractSweep = NearestNeighbourSweep(),
    solver: AbstractSolver = TimeIntegrationSolver(),
    multistarter: AbstractMultistart = LinearResponseMultistart(),
    response_measure: AbstractResponseMeasure = Demodulation(),
    precision: const.Precision = const.Precision.SINGLE,
) -> FrequencySweep:

    if isinstance(excitation, OneToneExcitation):
        if model.n_modes != len(excitation.modal_forces):
            raise ValueError("Number of modes in the model does not match the number of modal forces in the excitation.")

    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    model = model.to_dtype(dtype)
    excitation = excitation.to_dtype(dtype)
    sweeper = sweeper.to_dtype(dtype)
    multistarter = multistarter.to_dtype(dtype)
    
    solver.model = model
    solver.multistarter = multistarter

    frequency_sweep = solver.frequency_sweep(excitation, sweeper, response_measure)

    return frequency_sweep
