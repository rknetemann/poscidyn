# Frequency sweep

Use `poscidyn.frequency_sweep(...)` when you want continuation-like forward and backward response curves from many parallel time integrations.

## Required building blocks

A sweep combines four pieces:

- a `model`
- an `excitation`
- a `solver`
- a `response_measure`

If you do not specify them, Poscidyn defaults to:

- `TimeIntegration()`
- `LinearResponse()`
- `NearestNeighbour()`
- `Demodulation()`

## Minimal example

```python
import numpy as np
import poscidyn

omega_0 = np.array([1.0])
Q = np.array([80.0])
a = np.zeros((1, 1, 1))
b = np.zeros((1, 1, 1, 1))
b[0, 0, 0, 0] = 0.2

model = poscidyn.Nonlinear(omega_0=omega_0, Q=Q, a=a, b=b)

excitation = poscidyn.OneToneExcitation(
    drive_frequencies=np.linspace(0.8, 1.2, 200),
    drive_amplitudes=np.array([0.002, 0.004, 0.006]),
    modal_forces=np.array([1.0]),
)

solver = poscidyn.TimeIntegration(
    n_time_steps=100,
    max_steps=4096 * 20,
    rtol=1e-5,
    atol=1e-7,
    t_steady_state_factor=2.0,
)

response_measure = poscidyn.Demodulation(
    multiples=(1.0,),
    modal_contributions=np.array([1.0]),
)

result = poscidyn.frequency_sweep(
    model=model,
    excitation=excitation,
    solver=solver,
    response_measure=response_measure,
    precision=poscidyn.Precision.DOUBLE,
)
```

## Reading the result

The returned object stores modal and total responses separately. 

!!! note
    With `Demodulation`, the response arrays also retain the demodulated component axis. For a single `multiples=(1.0,)`, that axis has length 1.

Modal coordinates results:
```python
forward_amps = result.modal_coordinates.amplitudes["forward"]
backward_amps = result.modal_coordinates.amplitudes["backward"]
forward_phase = result.modal_coordinates.phases["forward"]
backward_phase = result.modal_coordinates.phases["backward"]
```

Modal superposition results:
```python
forward_amps = result.modal_superposition.amplitudes["forward"]
backward_amps = result.modal_superposition.amplitudes["backward"]
forward_phase = result.modal_superposition.phases["forward"]
backward_phase = result.modal_superposition.phases["backward"]
```

Stats:
```python
stats = result.stats
```


## Practical notes

- Increase `n_init_cond` in the multistart strategy if you expect multiple coexisting attractors.
- Choose `modal_contributions` in the response measure if you want the total response at a specific measurement point.
- Use `result.stats` to check how many trajectories completed successfully.
