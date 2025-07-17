import oscidyn
import numpy as np
import matplotlib.pyplot as plt

DRIVING_FREQUENCY = 2.0
DRIVING_AMPLITUDE = 1.5
INITIAL_DISPLACEMENT = np.zeros(1)
INITIAL_VELOCITY = np.zeros(1)

# time_response_steady_state = oscidyn.time_response(
#     model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
#     driving_frequency = DRIVING_FREQUENCY,
#     driving_amplitude = DRIVING_AMPLITUDE,
#     initial_displacement= INITIAL_DISPLACEMENT,
#     initial_velocity = INITIAL_VELOCITY,
#     solver = oscidyn.SteadyStateSolver(rtol=1e-3, atol=1e-8, n_time_steps=300, max_periods=1_000, max_steps=1_000_000),
# )
# time_steady_state,displacements_steady_state, velocities_steady_state = time_response_steady_state
# t_end_steady_state = time_steady_state[-1]

# time_response_standard = oscidyn.time_response(
#     model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
#     driving_frequency = DRIVING_FREQUENCY,
#     driving_amplitude = DRIVING_AMPLITUDE,
#     initial_displacement= INITIAL_DISPLACEMENT,
#     initial_velocity = INITIAL_VELOCITY,
#     solver = oscidyn.StandardSolver(t_end=int(t_end_steady_state*1.2), n_time_steps=10_000, max_steps=1_000_000),
# )
# time_standard, displacements_standard, velocities_standard = time_response_standard

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# axes[0].plot(time_steady_state, displacements_steady_state, label="SteadyStateSolver")
# axes[0].set_xlabel("Time")
# axes[0].set_ylabel("Displacement")
# axes[0].set_title("SteadyStateSolver Response")
# axes[0].grid(True)

# axes[1].plot(time_standard, displacements_standard, label="StandardSolver", color="orange")
# axes[1].set_xlabel("Time")
# axes[1].set_ylabel("Displacement")
# axes[1].set_title("StandardSolver Response")
# axes[1].grid(True)

# plt.tight_layout()
# plt.show()

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = np.linspace(0, 3.0, 10),
    driving_amplitudes = np.linspace(0.1, 1.5, 10),
    solver = oscidyn.SteadyStateSolver(rtol=1e-3, atol=1e-8, n_time_steps=50, max_steps=100_000),
)

