import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import oscidyn
import time

Q, omega_0, gamma = jnp.array([70000.0]), jnp.array([207.65e3*2*jnp.pi]), jnp.array([3e22])
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, gamma=gamma, omega_0=omega_0, x_ref=np.array([1e-8]))

MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(21, 1), linear_response_factor=1.5)
SOLVER = oscidyn.CollocationSolver(max_steps=1000, N_elements=128, K_polynomial_degree=2, max_iterations=10000, multistart=MULTISTART, verbose=True, rtol=1e-9, atol=1e-12, n_time_steps=500)
DRIVING_FREQUENCY = 207.65e3*2*jnp.pi + 1000
DRIVING_AMPLITUDE = 13.0
INITIAL_DISPLACEMENT = np.array([53.0])
INITIAL_VELOCITY = np.array([0.0])
PRECISION = oscidyn.Precision.DOUBLE

print("Time response: ", MODEL)
start_time = time.time()

time_response_steady_state = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = SOLVER,
    precision = PRECISION,
)

print("Time response completed in {:.2f} seconds".format(time.time() - start_time))

time_steady_state,displacements_steady_state, velocities_steady_state = time_response_steady_state
total_displacement_steady_state = displacements_steady_state.sum(axis=1)
total_velocity_steady_state = velocities_steady_state.sum(axis=1)

plt.figure()
plt.plot(time_steady_state, total_displacement_steady_state, label='Total Displacement')
plt.plot(time_steady_state, total_velocity_steady_state, label='Total Velocity')
plt.xlabel('Time')
plt.ylabel('Response')
plt.title(
    "Steady-State Time Response\n"
)
plt.legend()
plt.grid()
plt.show()