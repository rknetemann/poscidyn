import numpy as np
import matplotlib.pyplot as plt

# Time range
t = np.linspace(0, 2, 1000)  # 2 seconds of data with 1000 points

# Common frequency (in Hz)
frequency = 2  # 2 Hz

# Different amplitudes and phases
amplitude1 = 1.0
amplitude2 = 0.7
amplitude3 = 1.3

phase1 = 0            # 0 radians (0 degrees)
phase2 = np.pi/4      # π/4 radians (45 degrees)
phase3 = np.pi/2      # π/2 radians (90 degrees)

# Generate sinusoidal signals
angular_freq = 2 * np.pi * frequency
signal1 = amplitude1 * np.sin(angular_freq * t + phase1)
signal2 = amplitude2 * np.sin(angular_freq * t + phase2)
signal3 = amplitude3 * np.sin(angular_freq * t + phase3)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t, signal1, label=f'Amplitude={amplitude1}, Phase=0°')
plt.plot(t, signal2, label=f'Amplitude={amplitude2}, Phase=45°')
plt.plot(t, signal3, label=f'Amplitude={amplitude3}, Phase=90°')

# Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Three Sinusoids with Same Frequency but Different Amplitudes and Phases')
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
