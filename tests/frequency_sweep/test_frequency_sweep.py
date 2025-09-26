from jax import numpy as jnp
import oscidyn

Q, omega_0, gamma = 1000.0, 1.0, 0.00009
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.995, 1.005, 51)
DRIVING_AMPLITUDE = jnp.linspace(0.1*1/Q, 1.0*1/Q, 2)
#DRIVING_AMPLITUDE = jnp.array([1/Q])
SOLVER = oscidyn.MultipleShootingSolver(max_steps=500, m_segments=20, shooting_tolerance=1e-6, max_shooting_iterations=10, rtol=1e-6, atol=1e-7)
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