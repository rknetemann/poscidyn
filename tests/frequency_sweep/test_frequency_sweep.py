from jax import numpy as jnp
import oscidyn

Q, omega_0, gamma = 10.0, 1.0, 0.01
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 200)
DRIVING_AMPLITUDE = jnp.linspace(1.0*1/Q, 10.0*1/Q, 10)
SOLVER = oscidyn.MultipleShootingSolver(max_steps=100, m_segments=60, shooting_tolerance=1e-10, max_shooting_iterations=20, rtol=1e-9, atol=1e-12)
PRECISION = oscidyn.Precision.DOUBLE

print(MODEL)

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)