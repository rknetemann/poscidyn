# Define an oscillator

The supported built-in model is `NonlinearOscillator`. It represents modal
coordinates with linear damping and stiffness plus quadratic and cubic
restoring terms:

$$
\ddot q_i + \frac{\omega_{0,i}}{Q_i}\dot q_i + \omega_{0,i}^2 q_i
+ \sum_{j,k} a_{ijk} q_j q_k
+ \sum_{j,k,l} b_{ijkl} q_j q_k q_l = f_{e,i}(t).
$$

Construct it with one vector per mode and one tensor per restoring-force
order.

```python
import jax.numpy as jnp
import poscidyn

n_modes = 2
omega_0 = jnp.array([1.0, 1.7])
Q = jnp.array([80.0, 50.0])
a = jnp.zeros((n_modes, n_modes, n_modes))
b = jnp.zeros((n_modes, n_modes, n_modes, n_modes))
b = b.at[0, 0, 0, 0].set(0.2)

oscillator = poscidyn.NonlinearOscillator(
    omega_0=omega_0,
    Q=Q,
    a=a,
    b=b,
)
```

`omega_0` and `Q` must have shape `(n_modes,)`; `a` must have shape
`(n_modes, n_modes, n_modes)`; and `b` must have shape
`(n_modes, n_modes, n_modes, n_modes)`. The first index of `a` or `b` selects
the equation receiving the nonlinear force.

## Practical modelling notes

- Poscidyn assumes modal coordinates. Define your transformation and units
  before constructing the tensors.
- Supply zero tensors for terms that are absent; tensor shapes remain part of
  the model contract.
- The solver uses the model's settling-time estimate. For strongly nonlinear or
  multimodal systems, validate it by inspecting a time response and tune
  `t_steady_state_factor` if needed.
- A custom oscillator must implement the extension interface; see
  [Extending Poscidyn](../extending-poscidyn.md).
