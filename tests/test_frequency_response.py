from jax import numpy as jnp
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 2
DRIVING_FREQUENCY = jnp.linspace(0.1, 3.0, 2) # Shape: (1,)
DRIVING_AMPLITUDE = jnp.array([1.5, 0.3])  # Shape: (N_MODES,)

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.SteadyStateSolver(rtol=5e-2, atol=1e-8, n_time_steps=5000, max_windows=40, max_steps=100_000),
)

