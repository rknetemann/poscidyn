from jax import numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 1
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.2, 1000) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(0.01, 1.0, 100)  # Shape: (n_driving_amplitudes,)

# frequency_sweep = oscidyn.frequency_sweep(
#     model = MODEL,
#     sweep_direction = oscidyn.SweepDirection.FORWARD,
#     driving_frequencies = DRIVING_FREQUENCY,
#     driving_amplitudes = DRIVING_AMPLITUDE,
#     solver = oscidyn.SteadyStateSolver(ss_rtol=1e-2, ss_atol=1e-6, n_time_steps=500, max_windows=100, max_steps=4096),
# )

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = oscidyn.FixedTimeSteadyStateSolver(n_time_steps=300, max_steps=4096),
)

n_f = DRIVING_FREQUENCY.shape[0]
n_a = DRIVING_AMPLITUDE.shape[0]
amps = frequency_sweep.total_steady_state_displacement_amplitude.reshape(n_f, n_a)

# 2D Line plots
plt.figure()
for j in range(n_a):
    plt.plot(DRIVING_FREQUENCY, amps[:, j], label=f"A={DRIVING_AMPLITUDE[j]:.2g}")
plt.xlabel("Driving frequency")
plt.ylabel("Total steady-state displacement amplitude")
plt.legend(title="Drive amplitude")
plt.tight_layout()
plt.show()

# # 3D Surface plot
# from mpl_toolkits.mplot3d import Axes3D  # Required for projection='3d'

# # Create meshgrid with shape (n_f, n_a)
# FREQ, AMP = np.meshgrid(DRIVING_FREQUENCY, DRIVING_AMPLITUDE, indexing='ij')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(FREQ, AMP, amps, cmap='viridis', edgecolor='none')
# ax.set_xlabel("Driving frequency")
# ax.set_ylabel("Driving amplitude")
# ax.set_zlabel("Total steady-state displacement amplitude")
# fig.colorbar(surf, ax=ax, label='Displacement amplitude')
# plt.tight_layout()
# plt.show()
