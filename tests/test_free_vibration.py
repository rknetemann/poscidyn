import oscidyn
import numpy as np
import matplotlib.pyplot as plt

N_MODES = 1
DRIVING_FREQUENCY = 0
DRIVING_AMPLITUDE = np.array([0])  # Shape: (N_MODES,)
INITIAL_DISPLACEMENT = np.array([1]) # Shape: (N_MODES,)
INITIAL_VELOCITY = np.zeros(N_MODES) # Shape: (N_MODES,)

time_response_standard = oscidyn.time_response(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES),
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.StandardSolver(t_end=500, n_time_steps=5000, max_steps=1_000_000),
)
time_standard, displacements_standard, velocities_standard = time_response_standard
total_displacement_standard = displacements_standard.sum(axis=1)  # Sum across modes

fig, ax = plt.subplots(figsize=(10, 5))
# StandardSolver: individual modes + total
ax.plot(
    time_standard,
    total_displacement_standard,
    label="Total",
    color="black",
    linewidth=2
)
for i in range(displacements_standard.shape[1]):
    ax.plot(
        time_standard,
        displacements_standard[:, i],
        label=f"Mode {i+1}"
    )
ax.set_xlabel("Time")
ax.set_ylabel("Displacement")
ax.set_title("StandardSolver Response")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
