import numpy as np
import matplotlib.pyplot as plt
import oscidyn

Q, omega_0, gamma = np.array([10000.0]), np.array([1.0]), np.array([0.0])
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, gamma=gamma, omega_0=omega_0)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(21, 1), linear_response_factor=1.5)
SOLVER = oscidyn.CollocationSolver(max_steps=1000, N_elements=3, m_collocation_points=3, max_iterations=500, multistart=MULTISTART, verbose=True, rtol=1e-9, atol=1e-12, n_time_steps=500, throw=True)
DRIVING_FREQUENCY = omega_0*1.000
DRIVING_AMPLITUDE = 1.0 * omega_0**2/Q
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