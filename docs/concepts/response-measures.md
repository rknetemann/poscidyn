# RMS, minimum, and maximum

Response measures reduce the displacement history retained by a frequency
sweep to values that can be compared across initial conditions and frequency.
Pass one to `TimeIntegration(response_measure=...)`.

## Choose the measure for the question

- `Demodulation` is the usual choice for harmonic amplitude and phase at one
  or more drive-frequency multiples.
- `RMS` reports root-mean-square displacement. It is useful when phase is not
  meaningful or when a broadband scalar level is sufficient.
- `Min` and `Max` report the extrema across the retained time samples.

All measures produce modal values and a total value. The total is formed from
the modal displacement using `modal_contributions`; omit it to use unit weights
for every mode.

```python
measure = poscidyn.RMS(modal_contributions=jnp.array([1.0, 0.25]))
solver = poscidyn.TimeIntegration(
    oscillator=oscillator,
    excitation=excitation,
    response_measure=measure,
    n_time_steps=128,
)
```

`RMS`, `Min`, and `Max` do not estimate a phase. Their results are still
available through `result.forward.modal.amplitude` and
`result.forward.total.amplitude` so that plotting code can remain consistent.

For phase-resolved periodic responses, use [Demodulation](../getting-started/understanding-poscidyn/response-measures/demodulation.md).
