import poscidyn
import numpy as np

Q, omega_0, a, b = np.array([50.0, 50.0]), np.array([1.00, 2.00]), np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))
a[0,0,1] = 2.0
a[1,0,0] = 1.0
b[0,0,0,0] = 1.0
f_d = np.array([0.0144, 0.0144])
modal_contributions = np.array([1.0, 1.0])

omegas = np.linspace(0.9, 1.13, 256)
lambdas = np.linspace(0.1, 1.0, 8)

oscillator = poscidyn.Nonlinear(Q=Q, a=a, b=b, omega_0=omega_0)
excitation = poscidyn.DirectExcitation(f_d=f_d, omegas=omegas, lambdas=lambdas)
solver = poscidyn.TimeIntegration(rtol=1e-4, atol=1e-7, t_steady_state_factor=2.0)
response_measure = poscidyn.Demodulation(multiples=(1,), modal_contributions=modal_contributions)

frequency_sweep = poscidyn.frequency_sweep(
    oscillator = oscillator, excitation=excitation, solver=solver, 
    response_measure=response_measure, precision=poscidyn.Precision.DOUBLE
) 

print(frequency_sweep.stats)

import matplotlib.pyplot as plt

forward = np.asarray(frequency_sweep.modal_superposition.amplitudes["forward"])
backward = np.asarray(frequency_sweep.modal_superposition.amplitudes["backward"])

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(lambdas)))
fig, ax = plt.subplots(figsize=(9, 5.5))

for amp_idx, (drive_amp, color) in enumerate(zip(f_d*lambdas, colors)):
    label_forward = "Forward" if amp_idx == 0 else None
    label_backward = "Backward" if amp_idx == 0 else None
    ax.plot(
        omegas,
        forward[:, amp_idx],
        color=color,
        linewidth=1.8,
        label=label_forward,
    )
    ax.plot(
        omegas,
        backward[:, amp_idx],
        color=color,
        linestyle="--",
        linewidth=1.4,
        label=label_backward,
    )

ax.set_title("Minimum working frequency sweep example")
ax.set_xlabel("Drive frequency")
ax.set_ylabel("Response amplitude")
ax.grid(alpha=0.25)
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
plt.show()