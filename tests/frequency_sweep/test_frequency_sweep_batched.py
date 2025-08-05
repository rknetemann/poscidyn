from jax import numpy as jnp
import numpy as np
import sys
import os
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

DUFFING_COEFFICIENTS = jnp.linspace(-0.01, 0.03, 2)  # Shape: (n_duffing,)
DRIVING_FREQUENCIES = jnp.linspace(0.1, 2.0, 500) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDES = jnp.linspace(0.01, 1.0, 100)  # Shape: (n_driving_amplitudes,)

@jax.jit
def batched_frequency_sweep(
    duffing: float,
):
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1)
    model.gamma_hat.at[0, 0, 0, 0].set(duffing)
    
    return oscidyn.frequency_sweep(
        model=model,
        sweep_direction=oscidyn.SweepDirection.FORWARD,
        driving_frequencies=DRIVING_FREQUENCIES,
        driving_amplitudes=DRIVING_AMPLITUDES,
        solver=oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, rtol=1e-4, atol=1e-6),
    )

frequency_sweeps = jax.vmap(batched_frequency_sweep)(DUFFING_COEFFICIENTS)
