# Define an excitation

The supported harmonic drive is `DirectHarmonicExcitation`. It applies

$$
\mathbf f_e(t) = \mathbf f_d \odot \boldsymbol\lambda
\cos(\omega t),
$$

where `f_d` is the modal force vector and `lambdas` scales it elementwise.

```python
import jax.numpy as jnp
import poscidyn

excitation = poscidyn.DirectHarmonicExcitation(
    f_d=jnp.array([0.002, 0.0]),
    lambdas=jnp.array([1.0, 1.0]),
)
```

For `solver.frequency_sweep(omegas)`, the solver supplies each drive frequency
from `omegas`; leave `omega` unset. For `solver.time_response(...)`, provide
exactly one positive drive frequency when creating the excitation:

```python
excitation = poscidyn.DirectHarmonicExcitation(
    f_d=jnp.array([0.002]),
    omega=jnp.array([1.0]),
)
```

`f_d` and `lambdas` must be compatible with the number of oscillator modes.
Use a zero entry to leave a mode unforced.

`ParametricHarmonicExcitation` is present as development work, but its
end-to-end solver support is not part of the supported documentation contract.
See [status and roadmap](../../future-work.md) rather than relying on it for a
production workflow.
