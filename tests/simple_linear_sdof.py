import numpy as np
import poscidyn

n_modes = 1
Q = np.array([100])
omega_0 = np.array([1.0])
driving_frequency = np.linspace(0.5, 2.0, 500)
driving_amplitude = np.linspace(0.1, 1.0, 10)

model = poscidyn.NonlinearOscillator(n_modes=n_modes, Q=Q, omega_0=omega_0)