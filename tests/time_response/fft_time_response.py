import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
import oscidyn

Q, omega_0, gamma = 10.0, 1.0, 10.0
duration = 10000
n_time_steps = 20000
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
#MODEL = oscidyn.BaseDuffingOscillator(g1=jnp.array([omega_0/(Q)]), g2=jnp.array([omega_0]), g3=jnp.array([gamma]))
SOLVER = oscidyn.FixedTimeSolver(duration=duration, n_time_steps=n_time_steps, max_steps=4_096*20, rtol=1e-4, atol=1e-7)
#SOLVER = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1, rtol=1e-6, atol=1e-9, progress_bar=True)
DRIVING_FREQUENCY_1 = 0.999
DRIVING_FREQUENCY_2 = 1.001
DRIVING_AMPLITUDE = jnp.linspace(0.001, 1.0, 10)
INITIAL_DISPLACEMENT = np.array([0.0])
INITIAL_VELOCITY = np.array([0.0]) 

def calculate_fft(driving_amplitude):
    time_response_steady_state = oscidyn.time_response(
        model = MODEL,
        driving_frequency_1 = DRIVING_FREQUENCY_1,
        driving_frequency_2 = DRIVING_FREQUENCY_2,
        driving_amplitude = driving_amplitude,
        initial_displacement= INITIAL_DISPLACEMENT,
        initial_velocity = INITIAL_VELOCITY,
        solver = SOLVER,
    )

    # Unpack the results
    time_response_steady_state, displacements_steady_state, velocities_steady_state = time_response_steady_state

    # Cut off the first half to remove transients
    half = time_response_steady_state.shape[0] // 2
    time_response_steady_state = time_response_steady_state[half:]
    displacements_steady_state = displacements_steady_state[half:]
    velocities_steady_state = velocities_steady_state[half:]

    # Sum across modes
    total_displacement_steady_state = displacements_steady_state.sum(axis=1)

    frequency_vector = [0.998, 0.999, 1.001, 1.002]
    def discreteFourier(x, wk__, Dt):
        try:
            N = len(x)
            n = jnp.array(range(N))
            xk = np.zeros_like(wk__, dtype=complex)
            for i, wk in enumerate(wk__):
                xk[i] = np.sum(x[n] * np.exp(-1j * wk * n * Dt))
            return xk / (N / 2)
        except MemoryError:
            print('Memory Error')
            return []
        
    Dt = duration/n_time_steps
    X_f = discreteFourier(total_displacement_steady_state, frequency_vector, Dt)

    return X_f

calculate_fft_batch = jax.vmap(calculate_fft)(DRIVING_AMPLITUDE) # Shape: (n_drive_amplitudes, n_freq_components)



