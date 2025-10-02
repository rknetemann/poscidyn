import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"  # Use 95% of GPU memory

from jax import numpy as jnp
import oscidyn
import time

Q, omega_0, gamma = 10000.0, 1.0, 0.0002
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace((1.0-10*full_width_half_max), (1.0+10*full_width_half_max), 11) 
DRIVING_AMPLITUDE = jnp.linspace(0.1* omega_0**2/Q, 1.0*omega_0**2/Q, 2)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(11, 8), linear_response_factor=1.5)
SOLVER = oscidyn.MultipleShootingSolver(max_steps=1000, m_segments=5, max_shooting_iterations=500, rtol=1e-9, atol=1e-12, multistart=MULTISTART, verbose=True)
PRECISION = oscidyn.Precision.DOUBLE

print("Frequency sweeping: ", MODEL)

start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)

print("Frequency sweep completed in {:.2f} seconds".format(time.time() - start_time))

# ys: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes*2)
# x_max: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes)

drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = MULTISTART.generate_simulation_grid(
    MODEL, DRIVING_FREQUENCY, DRIVING_AMPLITUDE
)

oscidyn.plot_branch_exploration(
    drive_freq_mesh, drive_amp_mesh, frequency_sweep, tol_inside=1e-2
)

# 1. Doe een simulatie met Q=1e6, bepaal hoeveel frequency steps nodig zijn voor een goede resolutie (half width bandwidth)
# 2. Doe ik een grote simulatie of hak ik het in resolutie stukken (1000 punten vs meerdere 300)
# 3. Testen met het meegeven wan de ranges in de neural network

# CHRIS: 0.996 - 1.006, 300 punten