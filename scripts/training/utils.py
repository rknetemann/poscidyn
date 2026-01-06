from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
from typing import Any
import equinox as eqx

@eqx.filter_jit
def linear_response_amplitudes(
    x: PyTree[Float[Array, "..."]],
    kwargs: PyTree[Any, "..."],
) -> Float[Array, "n_freq"]:

    Q = x['Q'][:, jnp.newaxis]
    omega_0 = x['omega_0'][:, jnp.newaxis]

    f_omega = kwargs['f_omegas'][jnp.newaxis, :]
    f_amp = kwargs['f_amp'][:, jnp.newaxis]

    denom = (omega_0**2 - f_omega**2) + 1j * (f_omega * omega_0 / Q)
    X_modes = f_amp / denom # complex modal contributions
    X_total = jnp.sum(X_modes, axis=0) # complex superposition over modes
    return jnp.abs(X_total)