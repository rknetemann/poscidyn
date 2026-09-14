# 1-DOF Duffing time response

The [time-response quickstart](../../quickstart/time-response.md) is the
canonical runnable example. A time response is the right diagnostic before a
frequency sweep: it lets you inspect whether the transient estimate, sampling,
and response regime make sense for your parameter set.

Use `only_save_steady_state=False` while validating a new model. Plot
`response.displacement` and `response.velocity` to inspect settling and create
a phase portrait. Once the transient is credible, setting
`only_save_steady_state=True` reduces returned data to the final retained
window.

```python
response = solver.time_response(
    x0=jnp.array([0.0]),
    v0=jnp.array([0.0]),
    only_save_steady_state=False,
)
```

The solver estimates the transient from a linear model. Strong nonlinearities
can require a larger `t_steady_state_factor`; see
[accuracy, performance, and limitations](../../getting-started/limitations.md).
