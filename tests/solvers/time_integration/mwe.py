import poscidyn
import numpy as np

Q, omega_0, alpha, gamma = np.array([100.0]), np.array([1.00]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 2.55
modal_forces = np.array([1.0])

driving_frequency = np.linspace(0.9, 1.3, 501)
driving_amplitude = np.linspace(0.1, 1.0, 10)

MODEL = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL, excitor=EXCITOR,
) 
