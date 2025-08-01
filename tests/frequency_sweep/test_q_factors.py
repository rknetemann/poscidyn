import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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

    if Q == Q_FACTORS[0]:
        fig, axes = plt.subplots(len(Q_FACTORS), 1, sharex=True, figsize=(8, 3 * len(Q_FACTORS)))

    idx = Q_FACTORS.index(Q)
    ax = axes[idx]
    for i, A in enumerate(DRIVING_AMPLITUDE):
        ax.plot(frequency_sweep.frequencies, frequency_sweep.response[i], label=f"A = {A}")
    ax.set_title(f"Q = {Q}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Displacement")
    ax.legend()

    if Q == Q_FACTORS[-1]:
        plt.tight_layout()
        plt.show()
