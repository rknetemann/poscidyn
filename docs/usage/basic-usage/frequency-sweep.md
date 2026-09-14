# Frequency sweep

The solver-instance call is:

```python
result = solver.frequency_sweep(omegas)
```

`omegas` is a one-dimensional ordered array of positive drive frequencies. The
solver evaluates candidate trajectories from the configured multistart strategy
at every frequency and asks its configured synthetic-sweep method for selected
forward and backward paths.

Before calling the method, construct `TimeIntegration` with:

- an oscillator;
- a periodic excitation, normally `DirectHarmonicExcitation` without `omega`;
- a response measure;
- optional multistart and synthetic-sweep strategies;
- numerical settings such as `n_time_steps`, tolerances, and `max_steps`.

The [frequency-sweep quickstart](../../quickstart/frequency-sweep.md) is the
canonical complete example. Read [interpret sweep results](../../guides/interpreting-results.md)
before depending on a selected branch, and read
[limitations](../../getting-started/limitations.md) before treating it as a
physical quasi-static sweep.
