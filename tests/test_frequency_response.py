import oscidyn
import numpy as np
import matplotlib.pyplot as plt

time_response_steady_state = oscidyn.time_response(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
    driving_frequency = 2.0,
    driving_amplitude = 0.5,
    initial_displacement= np.zeros(1),
    initial_velocity = np.zeros(1),
    solver = oscidyn.SteadyStateSolver(rtol=1e-4, atol=1e-7, n_time_steps=5_000, max_periods=1_000, max_steps=1_000_000),
)

time_response_standard = oscidyn.time_response(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
    driving_frequency = 2.0,
    driving_amplitude = 0.5,
    initial_displacement= np.zeros(1),
    initial_velocity = np.zeros(1),
    solver = oscidyn.StandardSolver(t_end=500, n_time_steps=5_000, max_steps=100_000),
)

time,displacements, velocities = time_response_steady_state
time_2, displacements_2, velocities_2 = time_response_standard

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(time, displacements, label="SteadyStateSolver")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Displacement")
axes[0].set_title("SteadyStateSolver Response")
axes[0].grid(True)

axes[1].plot(time_2, displacements_2, label="StandardSolver", color="orange")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Displacement")
axes[1].set_title("StandardSolver Response")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# frequency_sweep = oscidyn.frequency_sweep(
#     model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
#     sweep_direction = oscidyn.SweepDirection.FORWARD,
#     driving_frequencies = np.linspace(0, 3.0, 10),
#     driving_amplitudes = np.linspace(0.1, 1.5, 10),
#     solver = oscidyn.SteadyStateSolver(rtol=1e-3, atol=1e-7, n_time_steps=50, max_steps=100_000),
# )

