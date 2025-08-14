import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = np.linspace(0.1, 1.5, 100)
DRIVING_AMPLITUDE = np.array([0.1, 1.0]) 

Q_FACTORS = [5, 10, 20, 50, 100]

times = []
total_displacements = []

for Q in Q_FACTORS:
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
    model.Q = Q
    print(f"Testing with Q = {Q}")
    
    frequency_sweep = oscidyn.frequency_sweep(
        model = model,
        sweep_direction = oscidyn.SweepDirection.FORWARD,
        driving_frequencies = DRIVING_FREQUENCY,
        driving_amplitudes = DRIVING_AMPLITUDE,
        #solver = oscidyn.FixedTimeSolver(t1=200, max_steps=4_096),
        solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096),
        #solver = oscidyn.SteadyStateSolver(rtol=1e-4, atol=1e-6, max_steps=4_096),
    )
