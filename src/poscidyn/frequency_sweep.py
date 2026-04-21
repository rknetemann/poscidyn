# Copyright (c) 2026 Raymond Knetemann
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
import jax
import jax.numpy as jnp

from .oscillator.abstract_oscillator import AbstractOscillator
from .solver.abstract_solver import AbstractSolver
from .excitation.abstract_excitation import AbstractExcitation
from .response_measure.abstract_response_measure import AbstractResponseMeasure
from .excitation.one_tone import OneToneExcitation
from .result.frequency_sweep import FrequencySweep 
from .solver.time_integration import TimeIntegration
from .response_measure.demodulation import Demodulation
from .multistart.abstract_multistart import AbstractMultistart
from .synthetic_sweep.abstract_synthetic_sweep import AbstractSyntheticSweep

from . import constants as const


def _to_dtype_if_supported(obj, dtype):
    if hasattr(obj, "to_dtype"):
        return obj.to_dtype(dtype)
    return obj

def frequency_sweep(
    oscillator: AbstractOscillator,
    excitation: AbstractExcitation,
    solver: AbstractSolver = TimeIntegration(),
    response_measure: AbstractResponseMeasure = Demodulation(),
    precision: const.Precision = const.Precision.SINGLE
) -> FrequencySweep:

    if isinstance(excitation, OneToneExcitation):
        if oscillator.n_modes != len(excitation.modal_forces):
            raise ValueError("Number of modes in the oscillator does not match the number of modal forces in the excitation.")

    if precision == const.Precision.DOUBLE:
        jax.config.update("jax_enable_x64", True)
        dtype = jnp.float64
    elif precision == const.Precision.SINGLE:
        jax.config.update("jax_enable_x64", False)
        dtype = jnp.float32
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    oscillator = _to_dtype_if_supported(oscillator, dtype)
    excitation = _to_dtype_if_supported(excitation, dtype)

    solver.oscillator = oscillator
    solver.excitation = excitation
    solver.response_measure = response_measure

    frequency_sweep = solver.frequency_sweep()

    return frequency_sweep
