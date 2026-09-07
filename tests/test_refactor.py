import numpy as np
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import poscidyn
import time
import sys

oscillator = poscidyn.oscillator.Nonlinear(omega_0=1.0, Q=10.0, a=0.0, b=1.0)
excitation = poscidyn.excitation.FreeVibration()

response_measure = poscidyn.response_measure.Demodulation()

solver = poscidyn.solver.TimeIntegration(oscillator=oscillator, excitation=excitation)