import matplotlib.pyplot as plt
import numpy as np
import poscidyn


def F_max(eta, omega_0, Q, b):
    """Estimate a forcing scale large enough to activate the Duffing nonlinearity."""
    return np.sqrt(
        4 * omega_0**6 / (3 * b * Q**2)
        * (eta + 1 / (2 * Q**2))
        * (1 + eta + 1 / (4 * Q**2))
    )


# Define system parameters.
Q = np.array([50.0, 50.0])
omega_0 = np.array([1.00, 2.00])
a = np.zeros((2, 2, 2))
b = np.zeros((2, 2, 2, 2))
a[0, 0, 1] = 2.0
a[1, 0, 0] = 1.0
b[0, 0, 0, 0] = 1.0

modal_forces = np.array([1.0, 1.0])
initial_displacement = np.array([0.0, 0.0])
initial_velocity = np.array([0.0, 0.0])

# Drive near the first-mode resonance so the quadratic coupling transfers
# energy into the second mode through the 1:2 internal resonance.
driving_frequency = np.array([1.00])
driving_amplitude = np.array([0.30 * F_max(0.3, omega_0[0], Q[0], b[0, 0, 0, 0])])

# Define classes.
model = poscidyn.NonlinearOscillator(Q=Q, a=a, b=b, omega_0=omega_0)
excitation = poscidyn.OneToneExcitation(
    drive_frequencies=driving_frequency,
    drive_amplitudes=driving_amplitude,
    modal_forces=modal_forces,
)
solver = poscidyn.TimeIntegrationSolver(
    max_steps=4096 * 12,
    n_time_steps=500,
    rtol=1e-5,
    atol=1e-7,
    t_steady_state_factor=1.0,
)

# Run the time response.
ts, xs, vs = poscidyn.time_response(
    model=model,
    excitation=excitation,
    initial_displacement=initial_displacement,
    initial_velocity=initial_velocity,
    solver=solver,
    precision=poscidyn.Precision.DOUBLE,
    only_save_steady_state=False,
)

# Plot displacement, velocity, and one phase portrait per mode.
t = np.asarray(ts)
x = np.asarray(xs)
v = np.asarray(vs)

fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0, 0].plot(t, x[:, 0], color="#1f77b4", linewidth=1.8, label="Mode 1")
axes[0, 0].plot(t, x[:, 1], color="#ff7f0e", linewidth=1.6, label="Mode 2")
axes[0, 0].set_title("Modal displacements")
axes[0, 0].set_ylabel("Displacement")
axes[0, 0].grid(alpha=0.25)
axes[0, 0].legend()

axes[0, 1].plot(t, v[:, 0], color="#d62728", linewidth=1.8, label="Mode 1")
axes[0, 1].plot(t, v[:, 1], color="#9467bd", linewidth=1.6, label="Mode 2")
axes[0, 1].set_title("Modal velocities")
axes[0, 1].set_ylabel("Velocity")
axes[0, 1].grid(alpha=0.25)
axes[0, 1].legend()

axes[1, 0].plot(x[:, 0], v[:, 0], color="#2ca02c", linewidth=1.5)
axes[1, 0].set_title("Phase portrait: mode 1")
axes[1, 0].set_xlabel("Displacement")
axes[1, 0].set_ylabel("Velocity")
axes[1, 0].grid(alpha=0.25)

axes[1, 1].plot(x[:, 1], v[:, 1], color="#8c564b", linewidth=1.5)
axes[1, 1].set_title("Phase portrait: mode 2")
axes[1, 1].set_xlabel("Displacement")
axes[1, 1].set_ylabel("Velocity")
axes[1, 1].grid(alpha=0.25)

fig.suptitle(
    f"Symmetry-breaking 1:2 internal resonance\n"
    f"Drive frequency = {driving_frequency[0]:.2f}, "
    f"drive amplitude = {driving_amplitude[0]:.4f}",
    y=0.98,
)
fig.tight_layout()
plt.show()
