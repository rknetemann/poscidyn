import oscidyn
import numpy as np
import matplotlib.pyplot as plt

N_MODES = 2
DRIVING_FREQUENCY = 2.0 # Shape: (1,)
DRIVING_AMPLITUDE = np.array([1.5, 0.3])  # Shape: (N_MODES,)

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.SteadyStateSolver(rtol=5e-2, atol=1e-8, n_time_steps=5000, max_periods=2048, max_steps=100_000),
)

