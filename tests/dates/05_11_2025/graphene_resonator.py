import numpy as np
import oscidyn

Q, omega_0, alpha, gamma = np.array([100.0, 200.0]), np.array([1.0, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
alpha[0,0,1] = 2 * 1.4542
alpha[1,0,0] = 1.4542
gamma[0,0,0,0] = 0.0638

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEP = oscidyn.NearestNeighbourSweep(sweep_direction=oscidyn.Forward())
EXCITATION = oscidyn.OneToneExcitation(drive_frequencies=np.linspace(0.8, 1.2, 151), drive_amplitudes=np.linspace(0.00005, 0.005, 10), 
                                       modal_forces=np.array([1.0, 0.0]))
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=0.7)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*10, verbose=True, throw=False, rtol=1e-3, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    excitation=EXCITATION,
    sweep = SWEEP,
    solver = SOLVER,
    multistart=MULTISTART,
    precision = PRECISION,
) 

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
            frequencies.extend(EXCITATION.drive_frequencies.tolist())
            responses.extend(frequency_sweep['max_x_total'][:, i_amp, i_disp, i_vel])
            colors.extend([EXCITATION.drive_amplitudes[i_amp]] * len(EXCITATION.drive_frequencies))

plt.figure(figsize=(12, 8))

# Plot total response
plt.subplot(n_modes + 1, 1, 1)
scatter = plt.scatter(frequencies, responses, c=colors, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Driving Amplitude")
plt.title(f"Total Response - {MODEL}")
plt.xlabel("Driving Frequency")
plt.ylabel("Response Amplitude")
plt.grid(True)

# Plot individual modes
for mode in range(n_modes):
    mode_responses = []
    for i_disp in range(n_init_disp):
        for i_vel in range(n_init_vel):
            for i_amp in range(n_amp):
                mode_responses.extend(max_x_modes[:, i_amp, i_disp, i_vel, mode])
    
    plt.subplot(n_modes + 1, 1, mode + 2)
    scatter = plt.scatter(frequencies, mode_responses, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Driving Amplitude")
    plt.title(f"Mode {mode + 1} Response")
    plt.xlabel("Driving Frequency")
    plt.ylabel("Response Amplitude")
    plt.grid(True)

plt.tight_layout()
plt.show()
