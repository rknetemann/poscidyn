import equinox as eqx
from jaxtyping import Array, Float
from typing import Any
import jax
import jax.numpy as jnp

from oscidyn.models import NonDimensionalisedModel, PhysicalModel
from oscidyn.solver import solve_rhs_batched

class FrequencyResponse(eqx.Module):
    excitation_frequency: jax.Array
    excitation_amplitude: jax.Array
    total_response: jax.Array
    modal_responses: jax.Array
    
def _get_steady_state(
    t: jax.Array,
    q: jax.Array,
    v: jax.Array
):
    discard_frac = 0.8

    return []

def frequency_response(
    model: PhysicalModel | NonDimensionalisedModel,
    excitation_frequency: jax.Array,
    excitation_amplitude: jax.Array,
    **solver_args: Any
) -> FrequencyResponse:
    n_modes = model.N
    n_freqs = 64
    n_amps = 5
    n_init_guesses = 50
    n_steps = 4096  
    
    # shape excitation_frequency_coarse: (n_freqs,)
    excitation_frequency_coarse = jnp.linspace(excitation_frequency[0], 
                                               excitation_frequency[-1], 
                                               n_freqs)
    print("excitation_frequency_coarse shape:", excitation_frequency_coarse.shape)
        
    # shape excitation_amplitude_coarse: (n_amps,)
    excitation_amplitude_coarse = jnp.linspace(excitation_amplitude[0],
                                               excitation_amplitude[-1],
                                               n_amps)
    print("excitation_amplitude_coarse shape:", excitation_amplitude_coarse.shape)
    
    # shape initial_displacement_coarse, initial_velocity_coarse: (n_modes, n_init_guesses)
    # shape initial_guesses_coarse: (n_modes, n_init_guesses * 2) ??
    initial_displacement_coarse = jnp.tile(jnp.linspace(0.0, 1.0, n_init_guesses), (n_modes, 1))
    initial_velocity_coarse = jnp.zeros((n_modes, n_init_guesses))
    initial_guesses_coarse = jnp.concatenate((initial_displacement_coarse, initial_velocity_coarse), axis=0)
    print("initial_displacement_coarse shape:", initial_displacement_coarse.shape)
    print("initial_velocity_coarse shape:", initial_velocity_coarse.shape)
    print("initial_guesses_coarse shape:", initial_guesses_coarse.shape)

    # shape x, v: (n_modes, n_freq, n_amp, n_init_guesses, n_steps):
    time_coarse, displacement_coarse, velocity_coarse = solve_rhs_batched(
        model,                         
        excitation_frequency_coarse,   
        excitation_amplitude_coarse,   
        initial_guesses_coarse,        
        1.0,                           
        n_steps,
        False,
    )

    print("time_coarse shape:", time_coarse.shape)
    print("displacement_coarse shape:", displacement_coarse.shape)
    print("velocity_coarse shape:", velocity_coarse.shape)

    # shape x, v: (n_modes, n_freq, n_amp, n_init_guesses):
    # This function will compute the steady state amplitude, velocities and phase.
    #ss_amplitude, ss_velocities, ss_phase = _get_steady_state(time_coarse, displacement_coarse, velocity_coarse)
    
    # shape x, v: (n_modes, n_freq, n_amp, n_max_solutions_per_freq):
    # Collect responses for each frequency and amplitude, so we reduce the number of initial guesses.
    #ss_amplitude, ss_velocities, ss_phase = _collect_responses()
    
    # shape x, v: (n_modes, n_freq, n_amp):
    # This function will remove values that would be observed in experiments.    
    #ss_amplitude, ss_velocities, ss_phase = _observed_in_experiment(ss_amplitude, ss_velocities, ss_phase, sweep_direction)

    # shape x, v: (n_freq, n_amp):
    # This function will sum the responses for each frequency and amplitude.
    #total_ss_amplitude, total_ss_velocities, total_ss_phase = _total_response(ss_amplitude, ss_velocities, ss_phase)

    return FrequencyResponse(
        excitation_frequency=excitation_frequency_coarse,
        excitation_amplitude=excitation_amplitude_coarse,
        total_response=jnp.zeros((n_freqs, n_amps)),
        modal_responses=jnp.zeros((n_modes, n_freqs, n_amps)),
    )

def plot_frequency_response(
    frequency_response: FrequencyResponse
):
    raise NotImplementedError(
        "Plotting frequency response is not implemented yet."
    )

