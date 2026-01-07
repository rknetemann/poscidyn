import poscidyn
import numpy as np

mu = np.array([1.5])
modal_forces = np.array([1.0])

driving_frequency = np.linspace(0.9, 1.3, 501)
driving_amplitude = np.linspace(0.1, 1.0, 10)

MODEL = poscidyn.VanDerPolOscillator(mu=mu)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL, excitor=EXCITOR,
) 
