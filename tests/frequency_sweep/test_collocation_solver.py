import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

from jax import numpy as jnp
import oscidyn
import time

import jax.profiler

x_ref = 1e-8
omega_ref = 207.65e3 * 2 * jnp.pi 
f = 13 
gamma = 3e22

f_hat = f / (omega_ref**2 * x_ref)
gamma_hat = gamma * (x_ref**2 / omega_ref**2)

Q, omega_0, gamma = 70000.0, 1.0, gamma_hat
full_width_half_max = omega_0 / Q


MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.997,1.004, 101) 
DRIVING_AMPLITUDE = jnp.linspace(0.1*f_hat, 1*f_hat, 2)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(21, 1), linear_response_factor=1.5)
SOLVER = oscidyn.CollocationSolver(max_steps=1000, N_elements=16, K_polynomial_degree=2, multistart=MULTISTART, max_iterations=300, verbose=True, rtol=1e-5, atol=1e-7, n_time_steps=500)
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

jax.profiler.save_device_memory_profile("memory_frequency_sweep.prof")

print("Frequency sweep completed in {:.2f} seconds".format(time.time() - start_time))

# ys: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes*2)
# x_max: (n_driving_frequencies, n_driving_amplitudes, n_initial_displacements, n_initial_velocities, n_modes)

drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = MULTISTART.generate_simulation_grid(
    MODEL, DRIVING_FREQUENCY, DRIVING_AMPLITUDE
)

title = f"Frequency sweep: Duffing (Q={Q}, $\\gamma$={gamma})"

oscidyn.plot_branch_exploration(
    drive_freq_mesh, drive_amp_mesh, frequency_sweep, tol_inside=1e-1, backbone={"f0": omega_0, "beta": gamma}, title=title
)

# 1. Doe een simulatie met Q=1e6, bepaal hoeveel frequency steps nodig zijn voor een goede resolutie (half width bandwidth)
# 2. Doe ik een grote simulatie of hak ik het in resolutie stukken (1000 punten vs meerdere 300)
# 3. Testen met het meegeven wan de ranges in de neural network

# CHRIS: 0.996 - 1.006, 300 punten