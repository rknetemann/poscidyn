import numpy as np
import matplotlib.pyplot as plt

omega_1 = 2 * np.pi * 1.0
omega_2 = 2 * np.pi * 2.0
omega_3 = 2 * np.pi * 3.5

fs = 1000.0
T = 10.0
N = int(fs * T)
t = np.arange(N) / fs

def signal(t, omega, phase):
    return np.cos(omega * t + phase)

def amplitude_at_freq_dft(x: np.ndarray, fs: float, f0: float, window: str | None = None):
    x = np.asarray(x)
    N = x.size
    n = np.arange(N)

    if window is None:
        w = np.ones(N)
    elif window == "hann":
        w = np.hanning(N)
    elif window == "hamming":
        w = np.hamming(N)
    else:
        raise ValueError("Unsupported window")

    exp_term = np.exp(-1j * 2*np.pi * f0 * n / fs)

    C = np.sum((x * w) * exp_term)

    cg = w.sum()
    A_peak = 2.0 * np.abs(C) / cg 
    phase_rad = np.angle(C)
    return A_peak, phase_rad

total_signal = 4*signal(t, omega_1, 0) + 1 * signal(t, omega_2, 2.2) + 2 * signal(t, omega_3, -1.2)

A_peak, phase_rad = amplitude_at_freq_dft(total_signal, fs, 3.5)
print(A_peak, phase_rad)

plt.figure()
plt.plot(t, total_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Sum of Cosine Signals')
plt.show()