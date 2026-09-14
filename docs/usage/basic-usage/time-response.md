# Time response

The solver-instance call is:

```python
response = solver.time_response(x0, v0, only_save_steady_state=False)
```

`x0` and `v0` are displacement and velocity vectors with one entry per mode.
The solver already owns the oscillator and excitation. For a periodic
excitation, its `omega` must contain exactly one positive frequency.

The returned `TimeResponse` object has:

- `time`, shape `(n_samples,)`;
- `displacement`, shape `(n_samples, n_modes)`;
- `velocity`, shape `(n_samples, n_modes)`.

Set `only_save_steady_state=True` to return only the retained periodic part of
an automatically chosen integration window. Alternatively, pass a positive
scalar `t` to integrate for an explicit duration; in that case
`only_save_steady_state` cannot be used.

See the complete [time-response quickstart](../../quickstart/time-response.md)
and [time-integration guide](../../getting-started/understanding-poscidyn/solvers/time-integration-solver.md).
