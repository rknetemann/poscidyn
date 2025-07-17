import oscidyn
import numpy as np
import matplotlib.pyplot as plt

N_MODES = 2
DRIVING_FREQUENCY = 2.0
DRIVING_AMPLITUDE = np.array([1.5, 0.3])  # Shape: (N_MODES,)
INITIAL_DISPLACEMENT = np.zeros(N_MODES) # Shape: (N_MODES,)
INITIAL_VELOCITY = np.zeros(N_MODES) # Shape: (N_MODES,)

time_response_steady_state = oscidyn.time_response(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.SteadyStateSolver(rtol=1e-3, atol=1e-8, n_time_steps=300, max_periods=1_000, max_steps=1_000_000),
)
time_steady_state,displacements_steady_state, velocities_steady_state = time_response_steady_state
total_displacement_steady_state = displacements_steady_state.sum(axis=1)  # Sum across modes
t_end_steady_state = time_steady_state[-1]

time_response_standard = oscidyn.time_response(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.StandardSolver(t_end=int(t_end_steady_state*1.2), n_time_steps=10_000, max_steps=1_000_000),
)
time_standard, displacements_standard, velocities_standard = time_response_standard
total_displacement_standard = displacements_standard.sum(axis=1)  # Sum across modes

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SteadyStateSolver: individual modes + total
axes[0].plot(time_steady_state,
             total_displacement_steady_state,
             label="Total",
             color="black",
             linewidth=2)
for i in range(displacements_steady_state.shape[1]):
    axes[0].plot(time_steady_state,
                 displacements_steady_state[:, i],
                 label=f"Mode {i+1}")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Displacement")
axes[0].set_title("SteadyStateSolver Response")
axes[0].grid(True)
axes[0].legend()

# StandardSolver: individual modes + total
axes[1].plot(time_standard,
             total_displacement_standard,
             label="Total",
             color="black",
             linewidth=2)
for i in range(displacements_standard.shape[1]):
    axes[1].plot(time_standard,
                 displacements_standard[:, i],
                 label=f"Mode {i+1}")
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Displacement")
axes[1].set_title("StandardSolver Response")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()
