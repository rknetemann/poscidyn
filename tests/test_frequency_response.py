from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 1
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.2, 100) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.1, 1.0, 100)  # Shape: (n_driving_amplitudes,)

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.SteadyStateSolver(ss_rtol=1e-2, ss_atol=1e-6, n_time_steps=500, max_windows=100, max_steps=4096),
)

# d = 0.01
# tau_d = - 2 * MODEL.Q * np.log(d * np.sqrt(1 - (1/MODEL.Q)**2) / DRIVING_FREQUENCY) 
# t_end = np.max(tau_d)
# print("Calculated t_end:", t_end)

# frequency_sweep = oscidyn.frequency_sweep(
#     model = MODEL,
#     sweep_direction = SWEEP_DIRECTION,
#     driving_frequencies = DRIVING_FREQUENCY,
#     driving_amplitudes = DRIVING_AMPLITUDE,
#     solver = oscidyn.FixedTimeSolver(t1=tau_d, n_time_steps=1000, max_steps=4096),
# )
