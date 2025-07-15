import jax.numpy as jnp

import oscidyn
import oscidyn.frequency_response

mdl = oscidyn.PhysicalModel.from_example(1).non_dimensionalise()

frequency_response_fw = oscidyn.frequency_response.frequency_response(
    model=mdl,
    excitation_frequency=jnp.array([0.5, 1.0, 1.5]),
    excitation_amplitude=jnp.array([1.0, 2.0])
)

initial_guesses = frequency_response_fw.initial_guesses


print("Initial guesses:", initial_guesses)