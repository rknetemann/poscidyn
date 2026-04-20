<h1 align='center'>Poscidyn</h1>
<h2 align='center'>Fast Simulation of Nonlinear Oscillator Dynamics in Python</h2>

Poscidyn (Python oscillator dynamics) is a Python toolkit based on [JAX](https://github.com/google/jax), designed to **streamline and accelerate time-response and frequency-sweep simulations**. It leverages novel parallelization strategies to gain a speed advantages over standard continuation software.

Features include:
- Frequency sweep simulation (forward and backward)
- Time-response simulation
- Built-in models of (nonlinear) oscillators
- Everything vmappable (batchable)
---

## Installation
```bash
pip install poscidyn[gpu]
```
Requires Python 3.10 or newer.

## Documentation
Have a look at our extensive documentation on how to install, use and extend this package: [https://rknetemann.github.io/poscidyn/](https://rknetemann.github.io/poscidyn/).

## Quick example

```python
import poscidyn
import numpy as np

Q, omega_0, a, b = np.array([50.0, 50.0]), np.array([1.00, 2.00]), np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))
a[0,0,1] = 2.0
a[1,0,0] = 1.0
b[0,0,0,0] = 1.0
modal_forces = np.array([1.0, 1.0])
modal_contributions = np.array([1.0, 1.0])

driving_frequency = np.linspace(0.9, 1.13, 256)
driving_amplitude = np.linspace(0.1, 1.0, 8) * 0.0144

model = poscidyn.NonlinearOscillator(Q=Q, a=a, b=b, omega_0=omega_0)
excitation = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
solver = poscidyn.TimeIntegrationSolver(max_steps=4096 * 20, n_time_steps=100, rtol=1e-5, atol=1e-7, t_steady_state_factor=2.0)
response_measure = poscidyn.Demodulation(multiples=(1,), modal_contributions=modal_contributions)

frequency_sweep = poscidyn.frequency_sweep(
    model = model, excitation=excitation, solver=solver, response_measure=response_measure, precision=poscidyn.Precision.DOUBLE
) 
```

![Frequency sweep](docs/images/symmetry_breaking_1_to_2_frequency_sweep.jpeg)

## Credits where they are due

[JAX](https://github.com/google/jax): a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

[Diffrax](https://github.com/patrick-kidger/diffrax): JAX-based library providing numerical differential equation solvers.

[Equinox](https://github.com/patrick-kidger/equinox): your one-stop JAX library, for everything you need that isn't already in core JAX.
