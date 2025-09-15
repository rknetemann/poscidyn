from jax import numpy as jnp
import time
import oscidyn

MODEL = oscidyn.DuffingOscillator(Q=jnp.array([100]), gamma=jnp.array([0.01]))
SWEEP_DIRECTION = oscidyn.SweepDirection.FORWARD
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 200) # Shape: (n_driving_frequencies,)
DRIVING_AMPLITUDE = jnp.linspace(1*1e-2, 100*1e-2, 10)  # Shape: (n_driving_amplitudes,)
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
plt.show()
