from jax import numpy as jnp
import oscidyn

Q, omega_0, gamma = 100000.0, 1.0, 0.0009
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.996, 1.004, 201)
DRIVING_AMPLITUDE = jnp.linspace(0.1*1/Q, 5.0*1/Q, 3)
SOLVER = oscidyn.MultipleShootingSolver(max_steps=50, m_segments=30, shooting_tolerance=1e-6, max_shooting_iterations=10, rtol=1e-5, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

print("Frequency sweeping: ", MODEL)

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweep_direction = SWEEP_DIRECTION,
    driving_frequencies = DRIVING_FREQUENCY,
    driving_amplitudes = DRIVING_AMPLITUDE,
    solver = SOLVER,
    precision = PRECISION,
)