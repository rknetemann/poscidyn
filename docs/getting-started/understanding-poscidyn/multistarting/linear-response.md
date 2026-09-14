# Configure multistart

`LinearResponse` creates `n_init_cond` random initial displacement and velocity
vectors for every requested frequency. Its sampling bounds are based on modal
linear-response scales:

$$
x_{\max,i} = \frac{Q_i}{\omega_{0,i}^2}c,
\qquad
v_{\max,i} = \frac{Q_i}{\omega_{0,i}}c,
$$

where `c` is `linear_response_factor`. The strategy samples uniformly inside
the resulting hyper-rectangle using `random_seed` for reproducibility.

```python
multistart = poscidyn.LinearResponse(
    n_init_cond=32,
    linear_response_factor=1.5,
    random_seed=0,
)
```

Increase `n_init_cond` when coexisting responses are likely. Increase
`linear_response_factor` when the expected nonlinear amplitude lies outside the
linear scale. Both increase the computational search burden, and a larger batch
also increases memory use.

These are heuristics for exploring basins of attraction, not a branch-complete
sampling guarantee. Validate sensitive results with changed seeds and denser
sampling.
