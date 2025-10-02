from jax import numpy as jnp
import oscidyn
import time

Q, omega_0, gamma = 10000.0, 1.0, 0.0004
full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace((1.0-10*full_width_half_max), (1.0+10*full_width_half_max), 201) 
DRIVING_AMPLITUDE = jnp.linspace(0.1* omega_0**2/Q, 1.0*omega_0**2/Q, 4)
# DRIVING_AMPLITUDE = jnp.array([1.0*omega_0**2/Q])
SOLVER = oscidyn.MultipleShootingSolver(max_steps=50, m_segments=10, max_shooting_iterations=50, rtol=1e-6, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

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

y0, y_max, mu = frequency_sweep
# y0: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes*2)
# y_max: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes)
# mu: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities)

drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = oscidyn.gen_grid_2(
    MODEL, DRIVING_FREQUENCY, DRIVING_AMPLITUDE
)

oscidyn.plot_branch_exploration(
    drive_freq_mesh, drive_amp_mesh, y_max, mu
)

# 1. Doe een simulatie met Q=1e6, bepaal hoeveel frequency steps nodig zijn voor een goede resolutie (half width bandwidth)
# 2. Doe ik een grote simulatie of hak ik het in resolutie stukken (1000 punten vs meerdere 300)
# 3. Testen met het meegeven wan de ranges in de neural network

# CHRIS: 0.996 - 1.006, 300 punten