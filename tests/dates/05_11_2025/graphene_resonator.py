import numpy as np
import oscidyn

Q, omega_0, alpha, gamma = np.array([100.0, 200.0]), np.array([1.0, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
alpha[0,0,1] = 2 * 1.4542
alpha[1,0,0] = 1.4542
gamma[0,0,0,0] = 0.0638

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = np.linspace(0.8, 1.2, 51)
DRIVING_AMPLITUDE = np.linspace(0.00005, 0.005, 10)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(23, 23), linear_response_factor=1.2)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*5, multistart=MULTISTART, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
) 


import matplotlib.pyplot as plt
n_freq, n_amp, n_init_disp, n_init_vel = frequency_sweep['max_x_total'].shape

frequencies = []
responses = []
colors = []




for i_disp in range(n_init_disp):
    for i_vel in range(n_init_vel):
        for i_amp in range(n_amp):
            frequencies.extend(DRIVING_FREQUENCY)
            responses.extend(frequency_sweep['max_x_total'][:, i_amp, i_disp, i_vel])
            colors.extend([DRIVING_AMPLITUDE[i_amp]] * len(DRIVING_FREQUENCY))

plt.figure(figsize=(10, 6))
scatter = plt.scatter(frequencies, responses, c=colors, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Driving Amplitude")
plt.title("Frequency Sweep")
plt.xlabel("Driving Frequency")
plt.ylabel("Response Amplitude")
plt.grid(True)
plt.show()

unsuccessful_frequencies = []
unsuccessful_amplitudes = []

for i_disp in range(n_init_disp):
    for i_vel in range(n_init_vel):
        for i_amp in range(n_amp):
            for i_freq in range(n_freq):
                if not frequency_sweep['successful'][i_freq, i_amp, i_disp, i_vel]:
                    unsuccessful_frequencies.append(DRIVING_FREQUENCY[i_freq])
                    unsuccessful_amplitudes.append(DRIVING_AMPLITUDE[i_amp])
                    
print(f"Number of unsuccessful points: {len(unsuccessful_frequencies)}")
print(f"Number of total points: {n_freq * n_amp * n_init_disp * n_init_vel}")
print(f"Percentage of unsuccessful points: {100.0 * len(unsuccessful_frequencies) / (n_freq * n_amp * n_init_disp * n_init_vel):.2f}%")

plt.figure(figsize=(10, 6))
plt.scatter(unsuccessful_frequencies, unsuccessful_amplitudes, color='red', alpha=0.7, label="Unsuccessful Points")
plt.title("Unsuccessful Frequency-Amplitude Combinations")
plt.xlabel("Driving Frequency")
plt.ylabel("Driving Amplitude")
plt.legend()
plt.grid(True)
plt.show()
