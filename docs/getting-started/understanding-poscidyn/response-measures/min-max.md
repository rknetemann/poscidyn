[<-- Go back to response measures](../start-here.md#response-measures)

# Min and max

`Min` and `Max` return the smallest or largest sampled value of the response over the supplied time window. They are purely time-domain extrema: useful for bounds, peak clipping checks, and asymmetric signals.

For each mode \(i\),

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

where \(x_i(t_k)\) is the modal displacement at sample \(t_k\).

If $\text{modal_contributions} = (\phi_1, \dots, \phi_{n_{\mathrm{modes}}})$ is provided, the total signal is reconstructed as

$$
x_{\mathrm{total}}(t_k)
=
\sum_{i=1}^{n_{\mathrm{modes}}}
\phi_i \, x_i(t_k)
$$

and the extrema are then

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

If `modal_contributions` is omitted, unit weights are used. When those weights come from a mode shape at a measurement point, the total entry becomes the physical minimum or maximum displacement at that point.

