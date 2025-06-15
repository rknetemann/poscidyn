import equinox as eqx
from jaxtyping import Array, Float

class FrequencyResponse(eqx.Module):
    excitation_type:  str
    excitation_frequency: Float[Array, ""] 
    excitation_amplitude: Float[Array, ""] 
    total_response: Float[Array, ""] 
    mode_response: Float[Array, ""]
    initial_guesses: Float[Array, ""] 
    
