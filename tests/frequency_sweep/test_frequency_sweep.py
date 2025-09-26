from jax import numpy as jnp
import time
import oscidyn

Q, omega_0, gamma = 200.0, 1.0, 0.1
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.9, 1.1, 200)
#DRIVING_AMPLITUDE = jnp.linspace(1*1/Q, 10*1/Q, 10)
DRIVING_AMPLITUDE = jnp.array([1.0/Q])
#SOLVER = oscidyn.SingleShootingSolver(max_steps=5000, rtol=1e-9, atol=1e-12)
SOLVER = oscidyn.MultipleShootingSolver(max_steps=5000, m_segments=8, shooting_tolerance=1e-10, max_shooting_iterations=20, rtol=1e-9, atol=1e-12)
PRECISION = oscidyn.Precision.DOUBLE

start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)

print("Frequency sweep completed in %.2f seconds." % (time.time() - start_time))
