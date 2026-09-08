import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import poscidyn

oscillator = poscidyn.oscillator.NonlinearOscillator(omega_0=np.array([1.0]), Q=np.array([10.0]), n_modes=1)
excitation = poscidyn.excitation.DirectHarmonicExcitation(f_d=np.array([1.0]), lambdas=np.array([1.0]))
response_measure = poscidyn.response_measure.Demodulation()
solver = poscidyn.solver.TimeIntegration(oscillator=oscillator, excitation=excitation, response_measure=response_measure)

ts, ys = solver.time_response(
    x0=np.array([0.0]),
    v0=np.array([0.0]),
    omega=1.0,
)
print(f"time response: ts shape={ts.shape}, ys shape={ys.shape}")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(ts, ys[:, 0], label="displacement")
plt.plot(ts, ys[:, 1], label="velocity")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Time response")
plt.legend()
plt.grid(alpha=0.25)
plt.show()


if __name__ == "__main__":
    omegas = np.linspace(0.5, 1.5, 100)
    sweep = solver.frequency_sweep(omegas=omegas)

    plt.plot(omegas, sweep.modal_superposition.amplitudes["forward"])
    plt.show()
