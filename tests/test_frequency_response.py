import oscidyn
import numpy as np

sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(N=1),
    driving_frequencies = np.linspace(0.1, 1, 100),
    driving_amplitudes = np.linspace(0.1, 1, 100),
    solver = oscidyn.StandardSolver(t_end=10.0),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
)