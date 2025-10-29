from jax import numpy as jnp
import time
import oscidyn

Q, omega_0, gamma = jnp.array([10.0, 8.0]), jnp.array([1.0, 2.0]), jnp.zeros((2,2,2,2))
gamma = gamma.at[0,0,0,0].set(0.02)


full_width_half_max = omega_0 / Q

MODEL = oscidyn.BaseDuffingOscillator.from_physical_params(Q=Q, gamma=gamma, omega_0=omega_0)
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
#DRIVING_FREQUENCY = jnp.linspace(1.0 - 25*full_width_half_max, 1.0 + 25*full_width_half_max, 101)
DRIVING_FREQUENCY = jnp.linspace(0.1, 3.0, 201)
DRIVING_AMPLITUDE = jnp.linspace(0.1 * omega_0[0]**2 / Q[0], 10.0 * omega_0[0]**2 / Q[0], 50)
SOLVER = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096*1000, rtol=1e-5, atol=1e-7, progress_bar=True)
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

duration = time.time() - start_time
print("Frequency sweep completed in %.2f seconds." % (duration))

tot_ss_disp_amp = frequency_sweep['tot_ss_disp_amp'].reshape(
    DRIVING_FREQUENCY.shape[0], -1
)  # Shape: (n_driving_frequencies, n_driving_amplitudes)

import matplotlib.pyplot as plt
min_amp = float(jnp.min(DRIVING_AMPLITUDE))
max_amp = float(jnp.max(DRIVING_AMPLITUDE))

plt.figure(figsize=(8, 6))
for i in range(DRIVING_AMPLITUDE.shape[0]):
    plt.plot(DRIVING_FREQUENCY, tot_ss_disp_amp[:, i])
plt.xlabel("Driving frequency")
plt.ylabel("Steady-state displacement amplitude")
plt.title(
    f"Frequency sweep: {duration:.2f} s"
)
plt.grid(True)
plt.tight_layout()
plt.savefig("frequency_sweep.png", dpi=300)
plt.show()
