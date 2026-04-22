# Excitation models

Poscidyn currently provides `DirectExcitation`, which defines a harmonic drive over one or more drive frequencies and drive amplitudes.

## `DirectExcitation`

The constructor is

```python
excitation = poscidyn.DirectExcitation(
    drive_frequencies,
    drive_amplitudes,
    modal_forces,
)
```

with:

- `drive_frequencies`: shape `(n_frequencies,)`
- `drive_amplitudes`: shape `(n_amplitudes,)`
- `modal_forces`: shape `(n_modes,)`

Internally, Poscidyn forms the outer product between `drive_amplitudes` and `modal_forces`, so each drive amplitude is applied with the same modal-force distribution.

## Example

```python
import numpy as np
import poscidyn

drive_frequencies = np.linspace(0.9, 1.1, 128)
drive_amplitudes = np.array([0.002, 0.004, 0.006])
modal_forces = np.array([1.0, 0.25])

excitation = poscidyn.DirectExcitation(
    drive_frequencies=drive_frequencies,
    drive_amplitudes=drive_amplitudes,
    modal_forces=modal_forces,
)
```

## How it is used

- In `poscidyn.time_response(...)`, you pass the same excitation object style, but currently with exactly one drive frequency and one drive amplitude.
- In `poscidyn.frequency_sweep(...)`, you pass an excitation object, typically `DirectExcitation`, so Poscidyn can generate the full sweep grid.
