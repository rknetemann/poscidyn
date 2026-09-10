[<-- Go back to response measures](../start-here.md#response-measures)

# L2

`L2` computes the root-mean-square (RMS) value of the response over the supplied time samples. It is a compact time-domain magnitude: useful when you want one number per signal but do not care about phase or frequency.

For each mode \(i\),

$$
A_i
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}
x_i(t_k)^2
}
$$

with \(x_i(t_k)\) the modal displacement at sample \(t_k\).

If $\text{modal_contributions} = (\phi_1, \dots, \phi_{n_{\mathrm{modes}}})$ is provided, the total signal is

$$
x_{\mathrm{total}}(t_k)
=
\sum_{i=1}^{n_{\mathrm{modes}}}
\phi_i \, x_i(t_k)
$$

and the total RMS is

$$
A_{\mathrm{total}}
=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}
x_{\mathrm{total}}(t_k)^2
}
$$

If `modal_contributions` is omitted, unit weights are used. When those weights come from a mode shape at a measurement point, the total entry becomes the physical RMS response at that point.
