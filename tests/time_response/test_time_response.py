import numpy as np
import matplotlib.pyplot as plt
import oscidyn

Q, omega_0, gamma = 5.0, 1.0, 0.4
MODEL = oscidyn.BaseDuffingOscillator(Q=np.array([Q]), gamma=np.array([gamma]), omega_0=np.array([omega_0]))
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=1000, max_steps=4096*5, rtol=1e-4, atol=1e-7, verbose=True)
DRIVING_FREQUENCY = 1.1
DRIVING_AMPLITUDE = 1.0*omega_0**2/Q
INITIAL_DISPLACEMENT = np.array([1.0])
INITIAL_VELOCITY = np.array([0])
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
