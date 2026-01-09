# Poscidyn in a nutshell

Poscidyn is a Python toolkit based on [JAX](https://github.com/google/jax), for simulating the dynamics of (nonlinear) oscillators using experimentally realistic setups, supporting both time- and frequency-domain analyses.

Features include:

- Frequency sweeping (forward and backward)
- Vmappable (batched) frequency sweeping 
- Built-in models of (nonlinear) oscillators

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

## Next steps

Have a look at the [Getting Started](./getting-started.md) page. 

## Credits where they are due

[JAX](https://github.com/google/jax): a Python library for accelerator-oriented array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning.

[Diffrax](https://github.com/patrick-kidger/diffrax): JAX-based library providing numerical differential equation solvers.

[Equinox](https://github.com/patrick-kidger/equinox): your one-stop JAX library, for everything you need that isn't already in core JAX.