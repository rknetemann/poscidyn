import numpy as np
import matplotlib.pyplot as plt
import oscidyn

# 1 mode: 
Q, omega_0, alpha, gamma = np.array([20.0]), np.array([1.0]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 1.0

# 2 modes:
Q, omega_0, alpha, gamma = np.array([10.0, 20.0]), np.array([1.00, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
gamma[0,0,0,0] = 0.0267
gamma[1,1,1,1] = 0.540


MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*5, n_time_steps=1000, rtol=1e-4, atol=1e-7, verbose=True)
DRIVING_FREQUENCY = 1.1
DRIVING_AMPLITUDE = 2.0
INITIAL_DISPLACEMENT = np.array([0.0, 0.0])
INITIAL_VELOCITY = np.array([0.0, 0.0])
PRECISION = oscidyn.Precision.SINGLE

time_response = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = SOLVER,
    precision = PRECISION,
    only_save_steady_state = True
)

ts, xs, vs = time_response
num_modes = xs.shape[-1]

# Plot total response
total_xs = xs.sum(axis=1)
total_vs = vs.sum(axis=1)

plt.figure(figsize=(10, 6))

# Use the same colormap as in the frequency sweep script
cmap = plt.cm.viridis

# Get one color per mode
mode_colors = cmap(np.linspace(0, 1, num_modes))

# Plot individual modes first
for mode in range(num_modes):
    color = mode_colors[mode]
    plt.subplot(num_modes + 1, 1, mode + 1)
    # same color, different linestyles for x and v
    plt.plot(ts, xs[:, mode], label='Displacement', color=color, linestyle='-')
    plt.plot(ts, vs[:, mode], label='Velocity',    color=color, linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title(f"Time Response (Mode {mode + 1})")
    plt.legend()
    plt.grid()

# Color for the total response (e.g. middle of colormap)
total_color = cmap(0.5)

# Plot total response at the end
plt.subplot(num_modes + 1, 1, num_modes + 1)
plt.plot(ts, total_xs, label='Displacement', color=total_color, linestyle='-')
plt.plot(ts, total_vs, label='Velocity',    color=total_color, linestyle='--')
plt.xlabel('Time')
plt.ylabel('Response')
plt.title("Time Response (Total)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# # Phase portrait: displacement vs velocity for each mode
# for mode in range(num_modes):
#     plt.figure(figsize=(8, 6))
#     plt.plot(xs[:, mode], vs[:, mode], label=f"Mode {mode + 1}")
#     plt.xlabel("Displacement")
#     plt.ylabel("Velocity")
#     plt.title(f"Phase Portrait (Mode {mode + 1})")
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

# Phase portrait for total response
# plt.figure(figsize=(8, 6))
# plt.plot(total_xs, total_vs, label="Total", linewidth=2, color="k")
# plt.xlabel("Displacement")
# plt.ylabel("Velocity")
# plt.title(f"Phase Space Time Response (F={DRIVING_AMPLITUDE})")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# # FFT of total response
# dt = ts[1] - ts[0]
# freqs = np.fft.rfftfreq(len(ts), dt)
# fft_total_xs = np.fft.rfft(total_xs)
# fft_total_vs = np.fft.rfft(total_vs)

# plt.figure(figsize=(10, 6))
# plt.subplot(2, 1, 1)
# plt.plot(freqs, np.abs(fft_total_xs), label='|X(f)|')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title("FFT of Total Displacement")
# plt.grid()

# plt.subplot(2, 1, 2)
# plt.plot(freqs, np.abs(fft_total_vs), label='|V(f)|', color='orange')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')
# plt.title("FFT of Total Velocity")
# plt.grid()

# plt.tight_layout()
# plt.show()
