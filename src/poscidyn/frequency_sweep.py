# Copyright (c) 2026 Raymond Knetemann
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
import jax
import jax.numpy as jnp

from .oscillator.abstract_oscillator import AbstractOscillator
from .solver.abstract_solver import AbstractSolver
from .excitation.abstract_excitation import AbstractExcitation
from .response_measure.abstract_response_measure import AbstractResponseMeasure
from .result.frequency_sweep import FrequencySweep 
from .solver.time_integration import TimeIntegration
from .response_measure.demodulation import Demodulation

from . import constants as const

def frequency_sweep(
    oscillator: AbstractOscillator,
    excitation: AbstractExcitation,
    solver: AbstractSolver = TimeIntegration(),
    response_measure: AbstractResponseMeasure = Demodulation(),
    precision: const.Precision = const.Precision.SINGLE
) -> FrequencySweep:

    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    solver.oscillator = oscillator
    solver.excitation = excitation
    solver.response_measure = response_measure

    frequency_sweep = solver.frequency_sweep()

    return frequency_sweep
