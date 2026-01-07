import numpy as np
import poscidyn

Q, omega_0, alpha, gamma = np.array([50.0, 23.0, 23.0]), np.array([1.0, 2.0, 3.0]), np.zeros((3,3,3)), np.zeros((3,3,3,3))
gamma[0,0,0,0] = 0.03
gamma[1,1,1,1] = 0.3
gamma[2,2,2,2] = 0.3
gamma[0,0,1,1] = 0.1
gamma[1,0,0,1] = 0.1

MODEL = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = poscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(0.01, 3.9, 351)
DRIVING_AMPLITUDE = np.linspace(0.1, 50.0, 5) * omega_0[0]**2/Q[0]
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(7, 7), linear_response_factor=50.2)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*20, multistart=MULTISTART, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
PRECISION = poscidyn.Precision.SINGLE

frequency_sweep = poscidyn.frequency_sweep(
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
