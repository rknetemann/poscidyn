import numpy as np
import matplotlib.pyplot as plt

import sys
import os


import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = 1.0
DRIVING_AMPLITUDE = 0.5 # (N_MODES,)
INITIAL_DISPLACEMENT = np.array([0.0]) # (N_MODES,)
INITIAL_VELOCITY = np.array([0.0]) # (N_MODES,)
MODEL = oscidyn.DuffingOscillator.from_example(n_modes=N_MODES)

time_response_steady_state = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.FixedTimeSolver(duration=200, n_time_steps=20000, max_steps=4_096*20, rtol=1e-12, atol=1e-14),
    #solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4096),
    #solver = oscidyn.SteadyStateSolver(max_steps=4096, rtol=1e-4, atol=1e-6)
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
