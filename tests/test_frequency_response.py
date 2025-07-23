from jax import numpy as jnp
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = jnp.linspace(0.1, 3.0, 500) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.1, 1.0, 10)  # Shape: (n_driving_amplitudes,)

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.SteadyStateSolver(ss_rtol=1e-3, ss_atol=1e-6, n_time_steps=1000, max_windows=100, max_steps=10_000),
)

