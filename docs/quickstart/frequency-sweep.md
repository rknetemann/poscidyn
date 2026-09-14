# First frequency sweep

This example computes forward and backward synthetic frequency-sweep curves
for a one-degree-of-freedom Duffing oscillator. It is the shortest complete
Poscidyn workflow: build the physical pieces, create one solver, then call its
method.

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import poscidyn

# q¨ + (omega_0 / Q) q˙ + omega_0² q + b q³ = f_d cos(omega t)
oscillator = poscidyn.NonlinearOscillator(
    omega_0=jnp.array([1.0]),
    Q=jnp.array([80.0]),
    a=jnp.zeros((1, 1, 1)),
    b=jnp.array([[[[0.2]]]]),
)
excitation = poscidyn.DirectHarmonicExcitation(
    f_d=jnp.array([0.002]),
    lambdas=jnp.array([1.0]),
)

solver = poscidyn.TimeIntegration(
    oscillator=oscillator,
    excitation=excitation,
    response_measure=poscidyn.Demodulation(
        multiples=(1.0,),
        modal_contributions=jnp.array([1.0]),
    ),
    multistart=poscidyn.LinearResponse(n_init_cond=8, random_seed=0),
    n_time_steps=64,
    periods_to_retain=2,
    max_steps=4096,
    rtol=1e-5,
    atol=1e-7,
)

result = solver.frequency_sweep(jnp.linspace(0.8, 1.2, 100))

frequency = result.frequency
forward = jnp.squeeze(result.forward.total.amplitude)
backward = jnp.squeeze(result.backward.total.amplitude)

plt.plot(frequency, forward, label="forward")
plt.plot(frequency, backward, "--", label="backward")
plt.xlabel("Drive frequency")
plt.ylabel("Response amplitude")
plt.legend()
plt.show()
```

`result.forward` and `result.backward` contain the selected synthetic-sweep
branches. `total` is the weighted modal superposition; `modal` retains an
individual result for each mode. With `Demodulation`, a singleton harmonic axis
remains in the output, hence `jnp.squeeze` in this one-mode, one-harmonic plot.

The calculation first integrates each frequency from several initial
conditions. It then selects a path through those candidates; it does not run a
traditional continuation algorithm. Read [the workflow](../concepts/workflow.md)
and [limitations](../getting-started/limitations.md) before changing numerical
settings or drawing branch-level conclusions.

Next: configure a [time response](time-response.md), or learn how to
[interpret sweep results](../guides/interpreting-results.md).
