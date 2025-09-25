import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp
import oscidyn

Q, omega_0, gamma = 200.0, 1.0, 0.01
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SOLVER = oscidyn.ShootingSolver(max_steps=500*5*5, rtol=1e-9, atol=1e-12, progress_bar=True)
DRIVING_FREQUENCY = 0.9
DRIVING_AMPLITUDE = 0.12
INITIAL_DISPLACEMENT = np.array([0.0])
INITIAL_VELOCITY = np.array([0.0])
PRECISION = oscidyn.Precision.DOUBLE

time_response_steady_state = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = SOLVER,
    precision = PRECISION,
)












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
