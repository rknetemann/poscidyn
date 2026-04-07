[<-- Go back to response measures](../start-here.md#response-measures)

# Demodulation

The `Demodulation` response measure extracts the amplitude and phase of the response at the drive frequency and at optional higher harmonics.

The raw output of the time integration consists of modal displacement time series \(q_i(t)\). However, experimental instruments such as lock-in amplifiers and vector network analyzers do not measure the full time signal directly. Instead, they return the amplitude and phase of the response at a specific frequency.

To make the simulated data directly comparable to such experimental observables, the time-domain response is demodulated.

## Definition

Demodulation is performed at selected multiples \(\mu_m\) of the drive frequency \(\omega_d\):

$$
\omega_{d,m} = \mu_m \omega_d.
$$

For each mode \(i\), the complex Fourier coefficient is computed as

$$
C_{m,i} = \sum_{n=1}^{N_t} w_n\, q_i(t_n)\, e^{-j \omega_{d,m} t_n},
$$

where:

- \(q_i(t_n)\) is the modal displacement of mode \(i\) at time \(t_n\),
- \(w_n\) is the window function,
- \(N_t\) is the number of time samples.

This operation corresponds to evaluating the discrete Fourier transform at the selected demodulation frequencies.

To convert the complex coefficient to a physical amplitude, the coherent gain of the window is computed as

$$
c_g = \sum_{n=1}^{N_t} w_n.
$$

The amplitude and phase of mode \(i\) at harmonic \(m\) are then

$$
A_{m,i} = \frac{2 \lvert C_{m,i} \rvert}{c_g},
\qquad
\phi_{m,i} = \arg(C_{m,i}).
$$

## Total response

If modal contributions \(\phi_i^{(\mathrm{shape})}\) are provided, the total displacement at the measurement point is constructed at the level of the complex coefficients:

$$
C_m^{\mathrm{total}} =
\sum_{i=1}^{n_{\mathrm{modes}}}
\phi_i^{(\mathrm{shape})} \, C_{m,i}.
$$

The corresponding total amplitude and phase are

$$
A_m^{\mathrm{total}} = \frac{2 \lvert C_m^{\mathrm{total}} \rvert}{c_g},
\qquad
\phi_m^{\mathrm{total}} = \arg(C_m^{\mathrm{total}}).
$$

## Parameters

### `multiples`
Sequence of frequency multiples \(\mu_m\).

- `(1.0,)` extracts the response at the drive frequency,
- `(1.0, 2.0, 3.0)` extracts higher harmonics,
- `(1/3,)` extracts a subharmonic response at \(\omega_d / 3\).

Both harmonic and subharmonic components can therefore be analyzed using the same demodulation framework.

This sequence must not be empty.

### `window`
Window function \(w_n\) applied prior to demodulation.

Supported values:

- `None` (rectangular window),
- `"hann"`,
- `"hamming"`.

### `modal_contributions`
Optional modal contributions \(\phi_i^{(\mathrm{shape})}\) used to reconstruct the total displacement.

## Returns

The result is a dictionary with `"modal"` and `"total"` entries.

### Modal response
- `"amplitude"`: \(A_{m,i}\), shape `(n_multiples, n_modes)`
- `"phase"`: \(\phi_{m,i}\), shape `(n_multiples, n_modes)`
- `"response_frequency"`: \(\omega_{d,m}\), shape `(n_multiples, n_modes)`

### Total response
- `"amplitude"`: \(A_m^{\mathrm{total}}\), shape `(n_multiples,)`
- `"phase"`: \(\phi_m^{\mathrm{total}}\), shape `(n_multiples,)`
- `"response_frequency"`: \(\omega_{d,m}\), shape `(n_multiples,)`
