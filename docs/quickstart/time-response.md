# First time response

Use `solver.time_response(...)` when you want the trajectory itself rather
than a reduced response measure. For a periodic excitation, provide exactly
one positive `omega` value on the excitation object.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import poscidyn

oscillator = poscidyn.NonlinearOscillator(
    omega_0=jnp.array([1.0]),
    Q=jnp.array([50.0]),
    a=jnp.zeros((1, 1, 1)),
    b=jnp.array([[[[0.2]]]]),
)
excitation = poscidyn.DirectHarmonicExcitation(
    f_d=jnp.array([0.002]),
    omega=jnp.array([1.0]),
)
solver = poscidyn.TimeIntegration(
    oscillator=oscillator,
    excitation=excitation,
    n_time_steps=200,
    periods_to_retain=2,
    max_steps=4096,
)

response = solver.time_response(
    x0=jnp.array([0.0]),
    v0=jnp.array([0.0]),
    only_save_steady_state=True,
)

plt.plot(response.time, response.displacement[:, 0])
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.show()
```

`response` is a `TimeResponse` object with `time`, `displacement`, and
`velocity` fields. It can also be unpacked as
`time, displacement, velocity = response`.

When `only_save_steady_state=True`, Poscidyn estimates and omits the transient
portion from the returned samples. That estimate is model-dependent; see the
[time-integration guide](../getting-started/understanding-poscidyn/solvers/time-integration-solver.md).
