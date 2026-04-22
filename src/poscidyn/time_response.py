# Copyright (c) 2026 Raymond Knetemann
# Licensed under the MIT License. See LICENSE file for details.

from __future__ import annotations
import jax
import jax.numpy as jnp
import time

from .oscillator.abstract_oscillator import AbstractOscillator
from .solver.abstract_solver import AbstractSolver
from .solver.time_integration import TimeIntegration
from .excitation.abstract_excitation import AbstractExcitation
from .excitation.direct import DirectExcitation
from . import constants as const

def time_response(
    model: AbstractOscillator,
    excitation: AbstractExcitation,
    initial_displacement: jax.Array, # Shape: (n_modes,)
    initial_velocity: jax.Array, # Shape: (n_modes,)
    solver: AbstractSolver = TimeIntegration(),
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

    initial_displacement = jnp.asarray(initial_displacement, dtype=dtype)
    initial_velocity = jnp.asarray(initial_velocity, dtype=dtype)
    
    if model.n_modes != initial_displacement.size:
        raise ValueError(f"Model has {model.n_modes} modes, but initial displacement has shape {initial_displacement.shape}. It should have shape ({model.n_modes},).")
    if model.n_modes != initial_velocity.size:
        raise ValueError(f"Model has {model.n_modes} modes, but initial velocity has shape {initial_velocity.shape}. It should have shape ({model.n_modes},).")
    
    model = model.to_dtype(dtype)
    excitation = excitation.to_dtype(dtype)
    solver.model = model

    driving_frequency = jnp.asarray(excitation.omegas, dtype=dtype)
    driving_amplitude = jnp.asarray(excitation.f_amps, dtype=dtype)

    if driving_frequency.ndim != 1 or driving_frequency.size != 1:
        raise ValueError(
            f"time_response requires a single drive frequency with shape (1,), got {driving_frequency.shape}."
        )
    if driving_amplitude.ndim != 2 or driving_amplitude.shape[0] != 1:
        raise ValueError(
            "time_response requires a single drive amplitude level so the resulting "
            f"modal forcing has shape (1, n_modes), got {driving_amplitude.shape}."
        )
    driving_amplitude = driving_amplitude[0]

    print("Time response: ", model)
    start_time = time.time()
    ts, ys = solver.time_response(driving_frequency, driving_amplitude, initial_displacement, initial_velocity, **kwargs)
    print("Time response completed in {:.2f} seconds".format(time.time() - start_time))


    xs = ys[:, :model.n_modes]  # Shape: (n_steps, n_modes)
    vs = ys[:, model.n_modes:]  # Shape: (n_steps, n_modes)

    return ts, xs, vs
