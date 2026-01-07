import numpy as np
import poscidyn

from zichao_parameters import Q, omega_0, alpha, gamma

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = poscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(0.1, 5.5, 401)
DRIVING_AMPLITUDE = np.linspace(0.01, 1.0, 5) * 0.03
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.2)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*10, multistart=MULTISTART, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
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
responses = []
colors = []

for i_disp in range(n_init_disp):
    for i_vel in range(n_init_vel):
        for i_amp in range(n_amp):
            frequencies.extend(DRIVING_FREQUENCY)
            responses.extend(frequency_sweep['max_x_total'][:, i_amp, i_disp, i_vel])
            colors.extend([DRIVING_AMPLITUDE[i_amp]] * len(DRIVING_FREQUENCY))

plt.figure(figsize=(12, 8))

# Plot total response
scatter = plt.scatter(frequencies, responses, c=colors, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Driving Amplitude")
plt.title(f"Total Response - {MODEL}")
plt.xlabel("Driving Frequency")
plt.ylabel("Response Amplitude")
plt.grid(True)
plt.show()