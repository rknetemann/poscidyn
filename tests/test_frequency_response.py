import oscidyn
import numpy as np

sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(N=1),
    driving_frequencies = np.linspace(0.1, 4, 100),
    driving_amplitudes = np.linspace(0.1, 1.5, 100),
    solver = oscidyn.StandardSolver(t_end=500.0, n_steps=10000, max_steps=80096),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
)