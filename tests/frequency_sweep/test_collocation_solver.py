import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import oscidyn
import numpy as np
import time
import matplotlib.pyplot as plt

Q, omega_0, gamma = np.array([10000.0]), np.array([1.0]), np.array([0.000])
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(1 - 100*full_width_half_max, 1 + 100*full_width_half_max, 32)
DRIVING_AMPLITUDE = np.linspace(0.1**omega_0**2/Q, 1**omega_0**2/Q, 2)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(5, 1), linear_response_factor=1.5)
SOLVER = oscidyn.CollocationSolver(max_steps=1000, N_elements=16, K_polynomial_degree=2, multistart=MULTISTART, max_iterations=1000, rtol=1e-9, atol=1e-12)
PRECISION = oscidyn.Precision.DOUBLE

print("Frequency sweeping: ", MODEL)

start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)

print("Frequency sweep completed in {:.2f} seconds".format(time.time() - start_time))

drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = MULTISTART.generate_simulation_grid(
    MODEL, DRIVING_FREQUENCY.flatten(), DRIVING_AMPLITUDE.flatten()
)

_, y_max, _ = frequency_sweep
y_max_reshaped = y_max.reshape(*drive_freq_mesh.shape, MODEL.n_modes)
fig, ax = plt.subplots()
for i_amp, amp in enumerate(DRIVING_AMPLITUDE):
    y_max_amp = y_max_reshaped[:, i_amp, ...]  # Shape: (F, D, V, n_modes)
    freq_amp = drive_freq_mesh[:, i_amp, ...]    # Shape: (F, D, V)
    ax.scatter(freq_amp.flatten(), y_max_amp[..., 0].flatten(), label=f'Amp={amp[0]:.2e}', s=5)

    ax.set_xlabel("Driving Frequency")
    ax.set_ylabel("Max Amplitude (Mode 1)")
    ax.set_title("Frequency Sweep")
    ax.legend()
    ax.grid(True)
plt.show()

# title = f"Frequency sweep: Duffing (Q={Q}, $\\gamma$={gamma})"

# oscidyn.plot_branch_exploration(
#     drive_freq_mesh, drive_amp_mesh, frequency_sweep, tol_inside=1e-1, backbone={"f0": omega_0, "beta": gamma}, title=title
# )

# CHRIS: 0.996 - 1.006, 300 punten