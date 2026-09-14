# Coupled Duffing systems

The one-degree-of-freedom quickstart is the best place to learn the workflow.
For coupled systems, the solver interface does not change: only array shapes
and non-zero tensor entries do.

```python
import jax.numpy as jnp
import poscidyn

a = jnp.zeros((2, 2, 2)).at[0, 0, 1].set(2.0).at[1, 0, 0].set(1.0)
b = jnp.zeros((2, 2, 2, 2)).at[0, 0, 0, 0].set(0.2)

oscillator = poscidyn.NonlinearOscillator(
    omega_0=jnp.array([1.0, 2.0]),
    Q=jnp.array([50.0, 50.0]),
    a=a,
    b=b,
)
excitation = poscidyn.DirectHarmonicExcitation(f_d=jnp.array([0.002, 0.0]))
solver = poscidyn.TimeIntegration(
    oscillator=oscillator,
    excitation=excitation,
    response_measure=poscidyn.Demodulation(
        multiples=(1.0,), modal_contributions=jnp.array([1.0, 1.0])
    ),
    multistart=poscidyn.LinearResponse(n_init_cond=16, random_seed=0),
    n_time_steps=128,
    periods_to_retain=2,
    max_steps=8192,
)
result = solver.frequency_sweep(jnp.linspace(0.85, 1.15, 160))
```

The entries `a[i, j, k]` and `b[i, j, k, l]` contribute respectively to the
quadratic and cubic restoring force of mode `i`. In a coupled system, inspect
the modal data before collapsing it into a total response.

See [define an oscillator](../usage/basic-usage/oscillator-models.md) for the
array conventions and [the workflow](../concepts/workflow.md) for interpretation
limits.
