import oscidyn
import numpy as np

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = np.linspace(0, 3.0, 150),
    driving_amplitudes = np.linspace(0.1, 1.5, 100),
    solver = oscidyn.SteadyStateSolver(tol=1e-1, max_steps=100_000),
)

