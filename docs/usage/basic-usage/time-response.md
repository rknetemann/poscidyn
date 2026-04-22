# Time response

Use `poscidyn.time_response(...)` when you want the simulated trajectory itself rather than a reduced response measure.

## Required inputs

The public function expects:

- `model`: an oscillator model
- `excitation`: currently a `DirectExcitation` with exactly one drive frequency and one drive amplitude
- `initial_displacement`: shape `(n_modes,)`
- `initial_velocity`: shape `(n_modes,)`
- `solver`: usually `TimeIntegration(...)`

## Minimal example

```python
import numpy as np
import poscidyn

omega_0 = np.array([1.0])
Q = np.array([100.0])
a = np.zeros((1, 1, 1))
b = np.zeros((1, 1, 1, 1))

model = poscidyn.Nonlinear(omega_0=omega_0, Q=Q, a=a, b=b)
excitation = poscidyn.DirectExcitation(
    drive_frequencies=np.array([1.0]),
    drive_amplitudes=np.array([0.01]),
    modal_forces=np.array([1.0]),
)
solver = poscidyn.TimeIntegration(
    n_time_steps=200,
    rtol=1e-5,
    atol=1e-7,
)

ts, xs, vs = poscidyn.time_response(
    model=model,
    excitation=excitation,
    initial_displacement=np.array([0.0]),
    initial_velocity=np.array([0.0]),
    solver=solver,
    precision=poscidyn.Precision.DOUBLE,
    only_save_steady_state=True,
)
```

## Return values

The function returns:

- `ts`: sampled times, shape `(n_steps,)`
- `xs`: modal displacements, shape `(n_steps, n_modes)`
- `vs`: modal velocities, shape `(n_steps, n_modes)`

## Practical notes

- `time_response(...)` now uses the same excitation object style as `frequency_sweep(...)`.
- The helper currently accepts only a single frequency and a single amplitude level. If you pass more, it raises a validation error instead of silently picking one.
- Set `only_save_steady_state=True` if you mainly care about the final periodic regime.
- If you need amplitude and phase instead of the raw trajectory, use `poscidyn.frequency_sweep(...)` together with a response measure such as `Demodulation`.
