import oscidyn
import numpy as np
import matplotlib.pyplot as plt

Q, omega_0, gamma = np.array([10000.0]), np.array([1.0]), np.array([0.0005])
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(1.0 - 25*full_width_half_max[0], 1.0 + 25*full_width_half_max[0], 201)
DRIVING_AMPLITUDE = np.linspace(0.2, 1.0, 2) * omega_0[0]**2/Q[0]
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(6, 1), linear_response_factor=1.5)
SOLVER = oscidyn.CollocationSolver(max_steps=1000, N_elements=128, m_collocation_points=4, multistart=MULTISTART, verbose=True, max_iterations=20, rtol=1e-9, atol=1e-12)
PRECISION = oscidyn.Precision.DOUBLE

frequency_sweep = oscidyn.frequency_sweep(
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
    y_max_amp = y_max_reshaped[:, i_amp, ...]  # Shape: (F, D, V, n_modes)
    freq_amp = drive_freq_mesh[:, i_amp, ...]    # Shape: (F, D, V)
    ax.scatter(freq_amp.flatten(), y_max_amp[..., 0].flatten(), label=f'Amp={amp:.2e}', s=5)

    ax.set_xlabel("Driving Frequency")
    ax.set_ylabel("Max Amplitude (Mode 1)")
    ax.set_title("Frequency Sweep")
    ax.legend()
    ax.grid(True)
plt.show()
