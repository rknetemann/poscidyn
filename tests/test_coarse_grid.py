Q, omega_0, gamma = 200.0, 1.0, 1.0

import oscidyn
from oscidyn.simulation.solvers.utils.coarse_grid import gen_coarse_grid
from jax import numpy as jnp

MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
DRIVE_FREQ = jnp.linspace(0.5, 1.5, 200)
DRIVE_AMP = jnp.linspace(1*1/Q, 10*1/Q, 10)

coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh, coarse_init_vel_mesh = gen_coarse_grid(MODEL, DRIVE_FREQ, DRIVE_AMP)
