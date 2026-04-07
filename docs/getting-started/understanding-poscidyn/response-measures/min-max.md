[<-- Go back to response measures](../start-here.md#response-measures)

# Min and max

The `Min` and `Max` response measures return the minimum or maximum value of the response over the provided time samples.

These measures are purely time-domain extrema. They do not perform any frequency-domain analysis, and they do not return a meaningful phase or response frequency.

## Definition

For each mode \(i\), the minimum response is defined as

$$
A_i^{\min}
=
\min_{k=1,\dots,N} x_i(t_k)
$$

and the maximum response is defined as

$$
A_i^{\max}
=
\max_{k=1,\dots,N} x_i(t_k)
$$

where \(x_i(t_k)\) is the modal response of mode \(i\) at time sample \(t_k\).

If a mode shape is provided, the total displacement is first reconstructed as

$$
x_{\mathrm{total}}(t_k)
=
\sum_{i=1}^{n_{\mathrm{modes}}}
\phi_i \, x_i(t_k)
$$

with \(\phi_i\) the modal contribution at the measurement point.

The total minimum and maximum are then

$$
A_{\mathrm{total}}^{\min}
=
\min_{k=1,\dots,N} x_{\mathrm{total}}(t_k)
$$

$$
A_{\mathrm{total}}^{\max}
=
\max_{k=1,\dots,N} x_{\mathrm{total}}(t_k)
$$

## Returned fields

Both `Min` and `Max` return the standard response-measure dictionary with `"modal"` and `"total"` entries.

For these measures:

- `"amplitude"` contains the minimum or maximum value,
- `"phase"` is set to `NaN`,
- `"response_frequency"` is set to `NaN`.

This is because extrema do not correspond to a unique phase or frequency.

## Modal and total response

### Modal response
For the modal part:

- `Min` returns the minimum of each modal coordinate over time,
- `Max` returns the maximum of each modal coordinate over time.

### Total response
For the total part:

- the modal responses are first combined using the mode shape,
- then the minimum or maximum of the reconstructed total displacement is returned.
