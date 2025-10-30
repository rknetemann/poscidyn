import numpy as np
import oscidyn

# 1 mode: 
Q, omega_0, alpha, gamma = np.array([20.0]), np.array([1.0]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 1.0
# 2 modes:
Q, omega_0, alpha, gamma = np.array([20.0, 17.0]), np.array([1.0, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
alpha[0,0,1] = 2 * 0.1
alpha[1, 0, 0] = 0.1
gamma[0,0,0,0] = 0.03

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(1.6, 2.4, 251)
DRIVING_AMPLITUDE = np.linspace(0.5, 7.0, 10) * omega_0[0]**2/Q[0]
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=7.2)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*5, multistart=MULTISTART, verbose=True, throw=True, rtol=1e-5, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
) #n_freq, n_amp, n_init_disp, n_init_vel

import matplotlib.pyplot as plt

n_freq, n_amp, n_init_disp, n_init_vel = frequency_sweep.shape

frequencies = []
responses = []
colors = []

for i_disp in range(n_init_disp):
    for i_vel in range(n_init_vel):
        for i_amp in range(n_amp):
            frequencies.extend(DRIVING_FREQUENCY)
            responses.extend(frequency_sweep[:, i_amp, i_disp, i_vel])
            colors.extend([DRIVING_AMPLITUDE[i_amp]] * len(DRIVING_FREQUENCY))

plt.figure(figsize=(10, 6))
scatter = plt.scatter(frequencies, responses, c=colors, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Driving Amplitude")
plt.title("Frequency Sweep")
plt.xlabel("Driving Frequency")
plt.ylabel("Response Amplitude")
plt.grid(True)
plt.show()
