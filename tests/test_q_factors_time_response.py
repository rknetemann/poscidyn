import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

EXTRA_TIME = 1.2

N_MODES = 1
DRIVING_FREQUENCY = 1.0
DRIVING_AMPLITUDE = 1.0  # Shape: (N_MODES,)
INITIAL_DISPLACEMENT = np.zeros(N_MODES) # Shape: (N_MODES,)
INITIAL_VELOCITY = np.zeros(N_MODES) # Shape: (N_MODES,)

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
    
    time_response_standard = oscidyn.time_response(
        model = model,  # Use the new model instance
        driving_frequency = DRIVING_FREQUENCY,
        driving_amplitude = DRIVING_AMPLITUDE,
        initial_displacement= INITIAL_DISPLACEMENT,
        initial_velocity = INITIAL_VELOCITY,
        solver = oscidyn.FixedTimeSolver(t1=500*EXTRA_TIME, n_time_steps=10_000, max_steps=1_000_000),
    )
    time_standard, displacements_standard, velocities_standard = time_response_standard
    times.append(time_standard)
    
    total_displacement_standard = displacements_standard.sum(axis=1) 
    total_displacements.append(total_displacement_standard)


plt.figure()
for t, total_disp, Q in zip(times, total_displacements, Q_FACTORS):
    plt.plot(t, total_disp, label=f"Q = {Q}")

plt.xlabel("Time")
plt.ylabel("Total displacement")
plt.legend()
plt.show()


