import poscidyn
import numpy as np

MODEL = poscidyn.NonlinearOscillator(n_modes=2)
MODEL.Q = np.array([50.0, 50.0])

print(MODEL)