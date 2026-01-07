import numpy as np
import poscidyn

from parameters_table_4 import Q, omega_0, alpha, gamma

MODEL = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = poscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(0.1, 4.5, 601)
DRIVING_AMPLITUDE = np.outer(np.linspace(0.15, 1.0, 6), np.array([0.9, 0.1, 0.05, 0.4, 0.03, 0.08, 0.1])) * 0.003
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*10, multistart=MULTISTART, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
PRECISION = poscidyn.Precision.SINGLE

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
) #n_freq, n_amp, n_init_disp, n_init_vel

max_x_total =  frequency_sweep['max_x_total']
max_x_modes = frequency_sweep['max_x_modes']

import matplotlib.pyplot as plt
n_freq, n_amp, n_init_disp, n_init_vel, n_modes = max_x_modes.shape

frequencies = []
responsivities = []
colors = []

for i_disp in range(n_init_disp):
    for i_vel in range(n_init_vel):
        for i_amp in range(n_amp):
            frequencies.extend(DRIVING_FREQUENCY)
            responses = frequency_sweep['max_x_total'][:, i_amp, i_disp, i_vel]
            responsivities.extend(responses / DRIVING_AMPLITUDE[i_amp, 0])
            colors.extend([DRIVING_AMPLITUDE[i_amp, 0]] * len(DRIVING_FREQUENCY))

plt.figure(figsize=(12, 12))

# Plot total responsivity
plt.subplot(2, 1, 1)
scatter = plt.scatter(frequencies, responsivities, c=colors, cmap='viridis', alpha=0.7, s=10)  # Decreased marker size
plt.colorbar(scatter, label="Driving Amplitude")
plt.title(f"Total Responsivity - {MODEL}")
plt.xlabel("Driving Frequency")
plt.ylabel("Responsivity")
plt.yscale('log')
plt.grid(True)

# Plot highest force individual modes as responsivity
plt.subplot(2, 1, 2)
colors = plt.cm.tab10(np.linspace(0, 1, n_modes))  # Generate distinct colors for each mode
for mode in range(n_modes):
    mode_responsivities = []
    for i_amp in range(n_amp):
        mode_responses = max_x_modes[:, i_amp, :, :, mode].max(axis=(1, 2))
        mode_responsivities.append(mode_responses / DRIVING_AMPLITUDE[i_amp, 0])
    mode_responsivities = np.array(mode_responsivities).max(axis=0)  # Only show the highest force
    plt.scatter(DRIVING_FREQUENCY, mode_responsivities, label=f"Mode {mode + 1}", alpha=0.7, color=colors[mode])
plt.title("Highest Force Individual Modes Responsivity")
plt.xlabel("Driving Frequency")
plt.ylabel("Responsivity")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot total responsivity in a separate figure
plt.figure(figsize=(12, 6))
scatter = plt.scatter(frequencies, responsivities, c=colors, cmap='viridis', alpha=0.7, s=10)  # Decreased marker size
plt.colorbar(scatter, label="Driving Amplitude")
plt.title(f"Total Responsivity - {MODEL}")
plt.xlabel("Driving Frequency")
plt.ylabel("Responsivity")
plt.yscale('log')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot highest force individual modes as responsivity in another figure
plt.figure(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, n_modes))  # Generate distinct colors for each mode
for mode in range(n_modes):
    mode_responsivities = []
    for i_amp in range(n_amp):
        mode_responses = max_x_modes[:, i_amp, :, :, mode].max(axis=(1, 2))
        mode_responsivities.append(mode_responses / DRIVING_AMPLITUDE[i_amp, 0])
    mode_responsivities = np.array(mode_responsivities).max(axis=0)  # Only show the highest force
    plt.scatter(DRIVING_FREQUENCY, mode_responsivities, label=f"Mode {mode + 1}", alpha=0.7, color=colors[mode])
plt.title("Highest Force Individual Modes Responsivity")
plt.xlabel("Driving Frequency")
plt.ylabel("Responsivity")
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
