<h1 align='center'>Poscidyn - Python oscillator dynamics</h1>
<h2 align='center'>Simulation and Visualization of Nonlinear Oscillator Dynamics</h2>

Poscidyn is a Python toolkit based on [JAX](https://github.com/google/jax), designed to **streamline and accelerate common workflows in nonlinear oscillator dynamics**. It enables the simulation and visualization of (nonlinear) oscillators using experimentally realistic setups, supporting both time- and frequency-domain analyses.

Features include:
- Built-in models of (nonlinear) oscillators
- Frequency sweeping (forward and backward)
- Everything vmappable
---

## Installation
```bash
pip install poscidyn
```
Requires Python 3.10 or newer.

## Documentation
Have a look at our extensive documentation on how to install, use and extend this package: [https://rknetemann.github.io/poscidyn/](https://rknetemann.github.io/poscidyn/).

## Quick example

```python
import poscidyn
import numpy as np

Q, omega_0, alpha, gamma = np.array([100.0]), np.array([1.00]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 2.55
modal_forces = np.array([1.0])

driving_frequency = np.linspace(0.9, 1.3, 501)
driving_amplitude = np.linspace(0.1, 1.0, 10)

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL, excitor=EXCITOR,
) 

```

## Credits where they are due

[JAX](https://github.com/google/jax): a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

[Diffrax](https://github.com/patrick-kidger/diffrax): JAX-based library providing numerical differential equation solvers.

[Equinox](https://github.com/patrick-kidger/equinox): your one-stop JAX library, for everything you need that isn't already in core JAX.
