import poscidyn
import numpy as np
import matplotlib.pyplot as plt

Q, omega_0, gamma = np.array([5.0]), np.array([1.0]), np.array([0.8])
full_width_half_max = omega_0 / Q

MODEL = poscidyn.NonlinearOscillator(Q=Q, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = poscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(1.0 - 25*full_width_half_max[0], 1.0 + 25*full_width_half_max[0], 61)
DRIVING_FREQUENCY = np.linspace(0.1, 2.0, 61)
DRIVING_AMPLITUDE = np.linspace(0.1, 1.0, 5) * omega_0[0]**2/Q[0]
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(6, 1), linear_response_factor=1.5)
SOLVER = poscidyn.CollocationSolver(max_steps=1000, N_elements=8, m_collocation_points=4, multistart=MULTISTART, verbose=True, max_iterations=200, rtol=1e-9, atol=1e-12)
PRECISION = poscidyn.Precision.DOUBLE

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)

drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = MULTISTART.generate_simulation_grid(
    MODEL, DRIVING_FREQUENCY.flatten(), DRIVING_AMPLITUDE.flatten()
)


y_max = frequency_sweep["y_max"]  # Shape: (F, A, D, V, n_modes)
y_max_reshaped = y_max.reshape(*drive_freq_mesh.shape, MODEL.n_modes)

fig, ax = plt.subplots()

for i_amp, amp in enumerate(DRIVING_AMPLITUDE):
    y_max_amp = y_max_reshaped[:, i_amp, ...]      # (F, D, V, n_modes)
    freq_amp = drive_freq_mesh[:, i_amp, ...]      # (F, D, V)
    ax.scatter(freq_amp.flatten(), y_max_amp[..., 0].flatten(),
               label=f'Amp={amp:.2e}', s=5)

# ---------- Backbone curve (angular frequency) ----------
# Build a smooth A-grid from 0 up to the largest amplitude seen.
A_max_seen = float(np.nanmax(y_max_reshaped[..., 0]))
A_grid = np.linspace(0.0, A_max_seen, 400)

# Ω_back(A) = sqrt(ω0^2 + (3/4) γ A^2)
omega0 = float(omega_0[0])
gamma_c = float(gamma[0])

omega_back_sq = omega0**2 + 0.75 * gamma_c * A_grid**2
# Keep only physically valid (real) frequencies
valid = omega_back_sq > 0
omega_back = np.sqrt(omega_back_sq[valid])
A_grid_valid = A_grid[valid]

ax.plot(omega_back, A_grid_valid, linewidth=2.0, label="Backbone (theory)")

# ---------- Plot cosmetics ----------
ax.set_xlabel("Driving frequency Ω (rad/s, normalized)")
ax.set_ylabel("Max displacement amplitude A (mode 1)")
ax.set_title("Frequency Sweep with Duffing Backbone")
ax.grid(True)
ax.legend()
plt.show()