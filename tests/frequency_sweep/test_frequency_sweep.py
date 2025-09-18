from jax import numpy as jnp
import time
import oscidyn

Q, omega_0, gamma = 200.0, 1.0, 1.0
MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=jnp.array([Q]), gamma=jnp.array([gamma]), omega_0=jnp.array([omega_0]))
MODEL = oscidyn.BaseDuffingOscillator(g1=jnp.array([omega_0/(Q)]), g2=jnp.array([omega_0]), g3=jnp.array([gamma]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.5, 1.5, 1000)
DRIVING_AMPLITUDE = jnp.linspace(1*1/Q, 10*1/Q, 10)
DRIVING_AMPLITUDE = jnp.array([0.12])
#DRIVING_AMPLITUDE = jnp.linspace(1.0e-2, 1.0e-1, 10)
SOLVER = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*200, rtol=1e-4, atol=1e-7, progress_bar=True)
PRECISION = oscidyn.Precision.SINGLE

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

tot_ss_disp_amp = frequency_sweep['tot_ss_disp_amp'].reshape(DRIVING_FREQUENCY.shape[0], DRIVING_AMPLITUDE.shape[0]) # Shape: (n_driving_frequencies, n_driving_amplitudes)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
for i in range(DRIVING_AMPLITUDE.shape[0]):
    plt.plot(DRIVING_FREQUENCY, tot_ss_disp_amp[:, i], label=f"A={DRIVING_AMPLITUDE[i]:.2f}")
plt.xlabel("Driving frequency")
plt.ylabel("Steady-state displacement amplitude")
plt.title("Frequency Sweep")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("frequency_sweep.png", dpi=300)
plt.show()

