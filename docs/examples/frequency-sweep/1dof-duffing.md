# 1-DOF Duffing sweep

The [frequency-sweep quickstart](../../quickstart/frequency-sweep.md) is the
canonical runnable 1-DOF Duffing example. This page highlights the modelling
and analysis choices that become important in a real study.

The scalar cubic coefficient `b[0, 0, 0, 0]` controls hardening when positive
and softening when negative. Sweep in the resonance region, retain enough
periods to resolve the desired harmonics, and increase `n_init_cond` if jumps
or coexisting attractors are plausible.

```python
b = jnp.array([[[[0.2]]]])
oscillator = poscidyn.NonlinearOscillator(
    omega_0=jnp.array([1.0]), Q=jnp.array([80.0]),
    a=jnp.zeros((1, 1, 1)), b=b,
)
```

When comparing forward and backward curves, first check
`result.stats["success_rate"]`. Then repeat a smaller frequency interval with
more initial conditions to determine whether a feature survives the multistart
sampling. A selected synthetic curve is evidence to investigate, not by itself
a complete bifurcation analysis.
