import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

EXTRA_TIME = 1.2

N_MODES = 1
DRIVING_FREQUENCY = np.linspace(0.1, 1.5, 100) # Shape: (100,)
DRIVING_AMPLITUDE = np.array([0.1, 1.0])  # Shape: (N_MODES,)

Q_FACTORS = [5, 10, 20, 50, 100]

def calculate_t_end(model, driving_frequency, d):
    tau_d = - 2 * model.Q * np.log(d * np.sqrt(1 - (1/model.Q)**2) / driving_frequency)
    return np.max(tau_d)
d = 0.1

times = []
total_displacements = []

for Q in Q_FACTORS:
    # Create a new model instance for each Q value instead of modifying the same model
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
    model.Q = Q
    print(f"Testing with Q = {Q}")

    t_end = calculate_t_end(model, DRIVING_FREQUENCY, d)
    print("Calculated t_end:", t_end)
    
    # frequency_sweep = oscidyn.frequency_sweep(
    #     model = model,  # Use the new model instance
    #     sweep_direction = oscidyn.SweepDirection.FORWARD,
    #     driving_frequencies = DRIVING_FREQUENCY,
    #     driving_amplitudes = DRIVING_AMPLITUDE,
    #     solver = oscidyn.FixedTimeSolver(t1=500*EXTRA_TIME, n_time_steps=10_000, max_steps=1_000_000),
    # )
    
    frequency_sweep = oscidyn.frequency_sweep(
        model = model,
        sweep_direction = oscidyn.SweepDirection.FORWARD,
        driving_frequencies = DRIVING_FREQUENCY,
        driving_amplitudes = DRIVING_AMPLITUDE,
        solver = oscidyn.SteadyStateSolver(ss_rtol=1e-2, ss_atol=1e-6, n_time_steps=500, max_windows=100, max_steps=4096),
    )


