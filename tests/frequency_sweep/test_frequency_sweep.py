from jax import numpy as jnp
import numpy as np
import sys
import os

import oscidyn

N_MODES = 1
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 200) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.1, 1.0, 10)  # Shape: (n_driving_amplitudes,)

import time
start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*100, rtol=1e-4, atol=1e-6, progress_bar=False),
    #solver = oscidyn.FixedTimeSolver(duration=1000.0, n_time_steps=512, rtol=1e-4, atol=1e-6),
)

print("Frequency sweep completed in %.2f seconds." % (time.time() - start_time))

tot_ss_disp_amp = frequency_sweep['tot_ss_disp_amp'].reshape(DRIVING_FREQUENCY.shape[0], DRIVING_AMPLITUDE.shape[0]) # Shape: (n_driving_frequencies, n_driving_amplitudes)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for i in range(DRIVING_AMPLITUDE.shape[0]):
    plt.plot(DRIVING_FREQUENCY, tot_ss_disp_amp[:, i], label=f"A={DRIVING_AMPLITUDE[i]:.2f}")
plt.xlabel("Driving frequency")
plt.ylabel("Steady-state displacement amplitude")
plt.title("Frequency Sweep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
