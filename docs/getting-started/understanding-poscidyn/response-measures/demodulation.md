[<-- Go back to response measures](../start-here.md#response-measures)

# Demodulation

`Demodulation` extracts a phasor from a steady-state time signal at selected multiples of the drive frequency. It is the response measure to use when you want quantities comparable to a lock-in amplifier or vector network analyzer: amplitude, phase, and response frequency.

The inputs are modal displacement time series \(x_i(t_n)\). Demodulation is applied directly to those modal coordinates and, optionally, to a weighted total response.

For each selected multiple \(\mu_m\), the demodulation frequency is

$$
\omega_m = \mu_m \omega_d,
$$

with \(\omega_d\) the drive angular frequency.

For each mode \(i\), the complex coefficient is

$$
C_{m,i} =
\sum_{n=1}^{N_t}
w_n \, x_i(t_n) \, e^{-j \omega_m t_n},
$$

where \(w_n\) is the chosen window. The window coherent gain is

$$
c_g = \sum_{n=1}^{N_t} w_n.
$$

Amplitude and phase then follow as

$$
A_{m,i} = \frac{2 |C_{m,i}|}{c_g},
\qquad
\phi_{m,i} = \arg(C_{m,i}).
$$


If you pass $\text{modal_contributions} = (\phi_1, \dots, \phi_{n_{\mathrm{modes}}})$, the total response is formed from the complex coefficients,

$$
C_m^{\mathrm{total}} =
\sum_{i=1}^{n_{\mathrm{modes}}}
\phi_i \, C_{m,i}.
$$

This is important: Poscidyn combines the modal coefficients before taking magnitude and phase, which preserves interference between modes.

If `modal_contributions` represents a mode-shape vector evaluated at a measurement point, the `"total"` block corresponds to the physical response at that point. If it is omitted, unit weights are used.

## Parameters

- `multiples`: non-empty sequence of frequency multiples. `(1.0,)` extracts the driven component, `(1.0, 2.0, 3.0)` adds superharmonics, and `(1/3,)` extracts a subharmonic.
- `window`: optional analysis window. Supported values are `None`, `"hann"`, and `"hamming"`.
- `modal_contributions`: optional 1D weight vector of length `n_modes` used to construct the total response.
