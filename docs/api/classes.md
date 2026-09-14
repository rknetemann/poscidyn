# Public API and result objects

This reference covers the supported solver-instance workflow. It deliberately
does not list experimental solver prototypes as ordinary public alternatives.

## Core components

### `NonlinearOscillator`

`poscidyn.NonlinearOscillator(omega_0, Q, a, b, n_modes=None)` defines the
modal nonlinear oscillator. See [define an oscillator](../usage/basic-usage/oscillator-models.md)
for shapes and the equation of motion.

### `DirectHarmonicExcitation`

`poscidyn.DirectHarmonicExcitation(f_d, omega=None, lambdas=jnp.array([1.0]))`
defines a modal single-tone harmonic drive. `omega` is required for a periodic
time response and supplied by `frequency_sweep` for a sweep.

### `TimeIntegration`

`poscidyn.TimeIntegration(oscillator, excitation=..., response_measure=..., ...)`
is the supported solver. Its principal methods are:

```python
response = solver.time_response(x0, v0, *, t=None, **options)
result = solver.frequency_sweep(omegas)
```

Relevant configuration parameters are `rtol`, `atol`, `n_time_steps`,
`max_steps`, `multistart`, `synthetic_sweep`, `t_steady_state_factor`,
`periods_to_retain`, `max_order_superharmonics`, and `throw`.

### Multistart and synthetic selection

`LinearResponse(n_init_cond=16, linear_response_factor=1.0, random_seed=0)`
creates the default initial-condition sampling strategy.

`NearestNeighbour(sweep_direction=[Forward(), Backward()], phase_weight=0.25,
seed_switch_penalty=0.05)` selects synthetic forward and backward paths from
the candidates.

### Response measures

- `Demodulation(multiples=(1.0,), window="hann", modal_contributions=None)`
- `RMS(modal_contributions=None)`
- `Min(modal_contributions=None)`
- `Max(modal_contributions=None)`

`modal_contributions` defines how modal coordinates form the total response.

## Result objects

`TimeResponse` provides `time`, `displacement`, and `velocity` fields and can
be unpacked into those three values.

`FrequencySweep` provides:

```python
result.frequency
result.forward.modal.amplitude
result.forward.total.amplitude
result.backward.modal.amplitude
result.backward.total.amplitude
result.stats
```

For `Demodulation`, branch responses also carry phase and response-frequency
data. See [interpret sweep results](../guides/interpreting-results.md) for
array shapes and validation guidance.

## API status

The old top-level `poscidyn.time_response(...)` and
`poscidyn.frequency_sweep(...)` helper style is not part of this documentation
contract. Use methods on `TimeIntegration`. Experimental shooting and
collocation code is described in [status and roadmap](../future-work.md), not
in this reference.
