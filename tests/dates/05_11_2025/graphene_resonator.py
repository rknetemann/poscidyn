import numpy as np
import oscidyn
import time

Q, omega_0, alpha, gamma = np.array([100.0, 200.0]), np.array([1.0, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
alpha[0,0,1] = 2 * 1.4542
alpha[1,0,0] = 1.4542
gamma[0,0,0,0] = 0.0638

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SWEEPER = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
EXCITOR = oscidyn.OneToneExcitation(drive_frequencies=np.linspace(0.8, 1.2, 151), drive_amplitudes=np.array([0.0005, 0.001, 0.003, 0.005]), 
                                       modal_forces=np.array([1.0, 0.0]))
MULTISTARTER = oscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*10, verbose=True, throw=False, rtol=1e-3, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    excitor = EXCITOR,
    sweeper = SWEEPER,
    solver = SOLVER,
    multistarter = MULTISTARTER,
    precision = PRECISION,
) 

end_time = time.time()
print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")

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
            frequencies.extend(EXCITOR.drive_frequencies.tolist())
            responses.extend(frequency_sweep['max_x_total'][:, i_amp, i_disp, i_vel])
            colors.extend([EXCITOR.drive_amplitudes[i_amp]] * len(EXCITOR.drive_frequencies))

PLOT_INDIVIDUAL_MODES = False  # Set to False to disable plotting individual modes

plt.figure(figsize=(12, 8))

if PLOT_INDIVIDUAL_MODES:
    # Plot total response with subplots for individual modes
    plt.subplot(n_modes + 1, 1, 1)
else:
    # Plot total response using the entire figure
    plt.subplot(1, 1, 1)

scatter = plt.scatter(frequencies, responses, c=colors, cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label="Driving Amplitude")
plt.title(f"Total Response - {MODEL}")
plt.xlabel("Driving Frequency")
plt.ylabel("Response Amplitude")
plt.grid(True)

# Add sweeped periodic solutions for forward and backward sweeps
sweeped_frequencies = EXCITOR.drive_frequencies
sweeped_solutions = frequency_sweep['sweeped_periodic_solutions']

if sweeped_solutions['forward'] is not None:
    for amp_idx, amp in enumerate(EXCITOR.drive_amplitudes):
        forward_responses = sweeped_solutions['forward'][:, amp_idx]
        plt.plot(sweeped_frequencies, forward_responses, label=f"Forward Sweep (Amp={amp:.3f})", linestyle='-', color='r')

if sweeped_solutions['backward'] is not None:
    for amp_idx, amp in enumerate(EXCITOR.drive_amplitudes):
        backward_responses = sweeped_solutions['backward'][:, amp_idx]
        plt.plot(sweeped_frequencies, backward_responses, label=f"Backward Sweep (Amp={amp:.3f})", linestyle='-', color='b')

plt.legend()

if PLOT_INDIVIDUAL_MODES:
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
