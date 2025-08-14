from jax import numpy as jnp
import numpy as np
import sys
import os

import oscidyn

N_MODES = 1
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 200) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.1, 1.0, 10)  # Shape: (n_driving_amplitudes,)

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, rtol=1e-4, atol=1e-6, progress_bar=False),
    #solver = oscidyn.FixedTimeSolver(duration=1000.0, n_time_steps=512, rtol=1e-4, atol=1e-6),
)

oscidyn.plot_frequency_sweep(frequency_sweep)

