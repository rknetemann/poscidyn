from jax import numpy as jnp
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn
import numpy as np

N_MODES = 1
MODEL = oscidyn.DuffingOscillator.from_example(n_modes=N_MODES)
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 500) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.01, 1.0, 100)  # Shape: (n_driving_amplitudes,)

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    #solver = oscidyn.FixedTimeSolver(t1=200, max_steps=4_096),
    solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, rtol=1e-4, atol=1e-6),
    #solver = oscidyn.SteadyStateSolver(rtol=1e-4, atol=1e-6, max_steps=4_096),
)

n_f = DRIVING_FREQUENCY.shape[0]
n_a = DRIVING_AMPLITUDE.shape[0]
amps = frequency_sweep.total_steady_state_displacement_amplitude.reshape(n_f, n_a)

# 2D Line plots in reversed grayscale

plt.figure()
# generate grayscale colors and reverse the sequence
gray_colors = plt.cm.gray(np.linspace(0.1, 0.9, n_a))[::-1]
for j, c in enumerate(gray_colors):
    plt.plot(DRIVING_FREQUENCY, amps[:, j], color=c, label=f"A={DRIVING_AMPLITUDE[j]:.2g}")
plt.xlabel("Driving frequency")
plt.ylabel("Total steady-state displacement amplitude")
#plt.legend(title="Drive amplitude")
plt.show()
