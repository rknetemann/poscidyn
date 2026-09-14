# Interpret sweep results

`solver.frequency_sweep(omegas)` returns a `FrequencySweep` object. Treat it
as the result of a candidate-search-and-selection procedure, not merely as two
arrays.

## The result layout

```python
result.frequency                 # one value per requested frequency
result.forward.modal.amplitude   # selected modal response
result.forward.total.amplitude   # selected weighted total response
result.backward.modal.amplitude
result.backward.total.amplitude
result.stats                     # trajectory completion summary
```

With `Demodulation(multiples=(1.0,))`, the modal amplitude typically has shape
`(n_frequency, 1, n_mode)` and the total amplitude has shape
`(n_frequency, 1)`. The middle dimension corresponds to requested harmonic
multiples. Preserve it when analysing multiple harmonics; remove it explicitly
only for a simple one-harmonic plot.

```python
amplitude = jnp.squeeze(result.forward.total.amplitude)
plt.plot(result.frequency, amplitude)
```

With `RMS`, `Min`, or `Max`, there is no demodulation-multiple dimension. Do
not write plotting code that assumes a fixed rank; inspect the chosen response
measure and the shape of the returned array.

## Validate before interpreting

Check `result.stats["success_rate"]`, `result.stats["n_successful"]`, and
`result.stats["n_total"]`. A smooth selected path does not repair failed
integrations or insufficient initial-condition coverage.

For a study where branch structure matters, repeat a representative region
with a larger `n_init_cond`, adjusted `linear_response_factor`, and appropriate
integration tolerances. Compare the selected paths rather than assuming that a
single configuration is definitive.

## Modal versus total response

`modal` describes the simulated modal coordinates. `total` is their weighted
superposition, using `modal_contributions` on the response measure. Use modal
data to diagnose energy transfer and internal resonance; use total data when a
specific measurement direction is the quantity of interest.
