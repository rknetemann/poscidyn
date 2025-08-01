from jax import numpy as jnp
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import oscidyn

N_MODES = 1
DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 1000)
DRIVING_AMPLITUDE = jnp.array([1])

gamma_hat_values = [0.00, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                    0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]  
frequency_sweep_results = []

for gamma_hat in gamma_hat_values:
    print(f"Testing with gamma_hat = {gamma_hat}")

    MODEL = oscidyn.NonlinearOscillator.from_example(n_modes=N_MODES)
    MODEL.gamma_hat = jnp.zeros((N_MODES, N_MODES, N_MODES, N_MODES)).at[0, 0, 0, 0].set(gamma_hat)

    frequency_sweep = oscidyn.frequency_sweep(
        model = MODEL,
        sweep_direction = oscidyn.SweepDirection.FORWARD,
        driving_frequencies = DRIVING_FREQUENCY,
        driving_amplitudes = DRIVING_AMPLITUDE,
        #solver = oscidyn.FixedTimeSolver(t1=200, max_steps=4_096),
        solver = oscidyn.FixedTimeSteadyStateSolver(max_steps=4_096),
        #solver = oscidyn.SteadyStateSolver(rtol=1e-4, atol=1e-6, max_steps=4_096),
    )
    
    frequency_sweep_results.append(frequency_sweep)

plt.figure(figsize=(10, 6))
for idx, frequency_sweep in enumerate(frequency_sweep_results):
    n_f = DRIVING_FREQUENCY.shape[0]
    n_a = DRIVING_AMPLITUDE.shape[0]
    amps = frequency_sweep.total_steady_state_displacement_amplitude.reshape(n_f, n_a)
    
    for j in range(n_a):
        plt.plot(DRIVING_FREQUENCY, amps[:, j], 
                 label=f"γ={gamma_hat_values[idx]:.3f}, A={DRIVING_AMPLITUDE[j]:.2g}")

plt.xlabel("Driving frequency")
plt.ylabel("Total steady-state displacement amplitude")
plt.legend(title="Parameters")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
