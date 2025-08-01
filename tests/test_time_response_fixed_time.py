import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = 1.356
DRIVING_AMPLITUDE = 1.0  # Shape: (N_MODES,)
INITIAL_DISPLACEMENT = np.array([0.0]) # Shape: (N_MODES,)
INITIAL_VELOCITY = np.array([0.0]) # Shape: (N_MODES,)
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
MODEL.Q = 10

d = 0.01
tau_d = - 2 * MODEL.Q * np.log(d * np.sqrt(1 - (1/MODEL.Q)**2) / DRIVING_FREQUENCY)
t_end = np.max(tau_d)
print("Calculated t_end:", t_end)


time_response_standard = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.FixedTimeSolver(t1=250, n_time_steps=10_000, max_steps=1_000_000),
)

# time_response_standard = oscidyn.time_response(
#     model = MODEL,
#     driving_frequency = DRIVING_FREQUENCY,
#     driving_amplitude = DRIVING_AMPLITUDE,
#     initial_displacement= INITIAL_DISPLACEMENT,
#     initial_velocity = INITIAL_VELOCITY,
#     solver = oscidyn.FixedTimeSteadyStateSolver(n_time_steps=500, max_steps=4_096*100),
# )

time_standard, displacements_standard, velocities_standard = time_response_standard
total_displacement_standard = displacements_standard.sum(axis=1)  # Sum across modes
total_velocity_standard = velocities_standard.sum(axis=1)  # Sum across modes

plt.figure()
plt.plot(time_standard, total_displacement_standard, label='Total Displacement')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs Time')
plt.grid(True)
plt.legend()
plt.show()

