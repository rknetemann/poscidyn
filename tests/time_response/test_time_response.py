import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = 1.0
DRIVING_AMPLITUDE = 1.0  # (N_MODES,)
INITIAL_DISPLACEMENT = np.zeros(N_MODES) # (N_MODES,)
INITIAL_VELOCITY = np.zeros(N_MODES) # (N_MODES,)
MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)

time_response_steady_state = oscidyn.time_response(
    model = MODEL,
    driving_frequency = DRIVING_FREQUENCY,
    driving_amplitude = DRIVING_AMPLITUDE,
    initial_displacement= INITIAL_DISPLACEMENT,
    initial_velocity = INITIAL_VELOCITY,
    solver = oscidyn.FixedTimeSolver(t1=200, max_steps=4_096),
    #solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4096),
    #solver = oscidyn.SteadyStateSolver(max_steps=4096, rtol=1e-4, atol=1e-6)
)
time_steady_state,displacements_steady_state, velocities_steady_state = time_response_steady_state
total_displacement_steady_state = displacements_steady_state.sum(axis=1)

plt.figure()
plt.plot(time_steady_state, total_displacement_steady_state, label='Total Displacement')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Steady-State Time Response')         
plt.legend()
plt.grid()
plt.show()
