import numpy as np
import matplotlib.pyplot as plt
import oscidyn

MODEL = oscidyn.DuffingOscillator(n_modes=1, Q=1000, gamma=0.010, omega_ref=1.0, x_ref=1.0)
SOLVER = oscidyn.FixedTimeSolver(duration=10000, n_time_steps=20000, max_steps=4_096*5, rtol=1e-4, atol=1e-7)
DRIVING_FREQUENCY = 1.0
DRIVING_AMPLITUDE = 1.0
INITIAL_DISPLACEMENT = np.array([0.0])
INITIAL_VELOCITY = np.array([0.0]) 

time_response_steady_state = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = SOLVER,
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
    f"Q = [{', '.join(f'{float(q):.2f}' for q in np.ravel(np.asarray(MODEL.Q)))}], "
    f"gamma = [{', '.join(f'{float(g):.2f}' for g in np.ravel(np.asarray(MODEL.alpha)))}], "
    f"f = {DRIVING_AMPLITUDE:.2f}"
)
plt.legend()
plt.grid()
plt.show()
