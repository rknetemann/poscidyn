# Oscillator models

Poscidyn currently exposes one oscillator class:

- `Nonlinear`: a modal oscillator model with linear, quadratic, and cubic terms.

For most structural-dynamics workflows, `Nonlinear` is the main entry point.

## `Nonlinear`

The model is defined in modal coordinates as

$$
\ddot q_i
+ \frac{\omega_{0,i}}{Q_i} \dot q_i
+ \omega_{0,i}^2 q_i
+ \sum_{j,k} a_{ijk} q_j q_k
+ \sum_{j,k,l} b_{ijkl} q_j q_k q_l
= f_i(t).
$$

Its constructor expects:

- `omega_0`: shape `(n_modes,)`, linear resonance frequencies.
- `Q`: shape `(n_modes,)`, quality factors.
- `a`: shape `(n_modes, n_modes, n_modes)`, quadratic coupling tensor.
- `b`: shape `(n_modes, n_modes, n_modes, n_modes)`, cubic coupling tensor.

If a term is absent, pass zeros of the correct shape.

### Single-mode example

```python
import numpy as np
import poscidyn

omega_0 = np.array([1.0])
Q = np.array([100.0])
a = np.zeros((1, 1, 1))
b = np.zeros((1, 1, 1, 1))
b[0, 0, 0, 0] = 0.1

model = poscidyn.Nonlinear(
    omega_0=omega_0,
    Q=Q,
    a=a,
    b=b,
)
```

### Two-mode example

```python
omega_0 = np.array([1.0, 2.0])
Q = np.array([80.0, 40.0])
a = np.zeros((2, 2, 2))
b = np.zeros((2, 2, 2, 2))

a[0, 0, 1] = 0.16
a[1, 0, 0] = 0.08
b[0, 0, 0, 0] = 1.0

model = poscidyn.Nonlinear(
    omega_0=omega_0,
    Q=Q,
    a=a,
    b=b,
)
```

## Practical note

Even for a single mode, Poscidyn uses arrays instead of scalars. That keeps the API consistent between single-mode and multi-mode problems.
