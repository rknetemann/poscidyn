import test_jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import time

# Set a key for reproducible random numbers
key = random.PRNGKey(0)

# Define parameters as individual values for realistic nanomechanical resonators
# Mode 1 parameters
m1 = 1e-15            # Mass in femtograms (10^-15 kg)
c1 = 1e-11            # Linear damping (corresponds to Q ≈ 10^4)
k1 = 1.0              # Linear stiffness (N/m)
alpha1 = 1e8          # Cubic stiffness of mode 1
beta1 = 1e16          # Quintic stiffness of mode 1
gamma1 = 1e-6         # Nonlinear damping of mode 1

# Mode 2 parameters
m2 = 1.2e-15          # Mass of mode 2aimport jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define the coupled oscillator model including Duffing nonlinearity, nonlinear damping,
# and a linear mode-coupling term.
def coupled_oscillator(state, t, f, omega, params):
    # state = [x1, v1, x2, v2]
    x1, v1, x2, v2 = state
    # Unpack parameters
    zeta1 = params['zeta1']
    alpha1 = params['alpha1']
    eta1 = params['eta1']
    zeta2 = params['zeta2']
    r = params['r']         # effective stiffness (or squared frequency) for mode 2
    alpha2 = params['alpha2']
    eta2 = params['eta2']
    k_c = params['k_c']     # coupling constant

    dx1dt = v1
    dv1dt = f * jnp.cos(omega * t) - 2 * zeta1 * v1 - x1 - alpha1 * x1**3 - eta1 * x1**2 * v1 - k_c * (x1 - x2)
    dx2dt = v2
    dv2dt = - 2 * zeta2 * v2 - r * x2 - alpha2 * x2**3 - eta2 * x2**2 * v2 - k_c * (x2 - x1)
    return jnp.array([dx1dt, dv1dt, dx2dt, dv2dt])

# A 4th-order Runge-Kutta step for the coupled system.
@test_jax.jit
def rk4_step_coupled(state, t, dt, f, omega, params):
    k1 = coupled_oscillator(state, t, f, omega, params)
    k2 = coupled_oscillator(state + dt/2 * k1, t + dt/2, f, omega, params)
    k3 = coupled_oscillator(state + dt/2 * k2, t + dt/2, f, omega, params)
    k4 = coupled_oscillator(state + dt * k3, t + dt, f, omega, params)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# Simulate the time series for the coupled system.
def simulate_time_series_coupled(f, omega, params, dt=0.01, n_time_steps=8000):
    # initial state: both modes start at rest
    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    t0 = 0.0

    def step_fn(carry, _):
        state, t = carry
        new_state = rk4_step_coupled(state, t, dt, f, omega, params)
        new_t = t + dt
        # record only the displacement of the primary mode x1
        return (new_state, new_t), new_state[0]
    
    (final_state, final_t), x1_series = test_jax.lax.scan(step_fn, (state, t0), None, length=n_time_steps)
    return x1_series

# Compute the steady-state amplitude by discarding a fraction of the time series as transient.
def compute_amplitude_coupled(f, omega, params, dt=0.01, n_time_steps=8000, transient_fraction=0.5):
    x1_series = simulate_time_series_coupled(f, omega, params, dt=dt, n_time_steps=n_time_steps)
    transient_cut = int(n_time_steps * transient_fraction)
    steady_state = x1_series[transient_cut:]
    # Return the maximum absolute displacement in the steady state
    amp = jnp.max(jnp.abs(steady_state))
    return amp

# Main simulation parameters (dimensionless)
# Parameters for mode 1 and mode 2 (typical literature–like values)
coupled_params = {
    'zeta1': 0.02,
    'alpha1': 1.0,
    'eta1': 0.1,
    'zeta2': 0.015,
    'r': 1.1,         # slightly shifted resonance for mode 2
    'alpha2': 0.8,
    'eta2': 0.05,
    'k_c': 0.2,       # coupling strength between modes
}

# Frequency sweep: we sweep the (dimensionless) drive frequency near the resonance.
frequencies = jnp.linspace(0.8, 1.2, 100)

# List of different driving force amplitudes to see the effect of nonlinearity.
drive_amplitudes = [0.1, 0.3, 0.5, 0.7]

# Settings for the time integration
dt = 0.01           # time step
n_time_steps = 8000      # total number of time steps per frequency
transient_fraction = 0.5  # discard the first half as transient

# For each drive amplitude, compute the steady-state amplitude vs drive frequency.
results = {}
for f in drive_amplitudes:
    # Vectorize the computation over frequency.
    compute_for_freq = test_jax.jit(test_jax.vmap(
        lambda omega: compute_amplitude_coupled(f, omega, coupled_params, dt, n_time_steps, transient_fraction)
    ))
    amps = compute_for_freq(frequencies)
    results[f] = amps

# Convert JAX arrays to NumPy for plotting.
frequencies_np = jnp.array(frequencies).tolist()

plt.figure(figsize=(8, 5))
for f in drive_amplitudes:
    amps_np = jnp.array(results[f])
    plt.plot(frequencies_np, amps_np, label=f'Drive amplitude = {f:.2f}')
plt.xlabel("Drive frequency (dimensionless)")
plt.ylabel("Steady-state amplitude (x₁)")
plt.title("Dynamic Frequency Response of a Nonlinear Coupled Resonator")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

c2 = 1.2e-11          # Linear damping for mode 2
k2 = 1.1              # Linear stiffness of mode 2
alpha2 = 1.2e8        # Cubic stiffness of mode 2
beta2 = 1.2e16        # Quintic stiffness of mode 2
gamma2 = 1.2e-6       # Nonlinear damping of mode 2

# Coupling parameters
kc = 0.05             # Linear coupling strength
delta = 1e6           # Nonlinear coupling strength

# Driving parameters
F1 = 1e-12            # Driving amplitude for mode 1 (pN)
F2 = 5e-13            # Driving amplitude for mode 2 (pN)
h1 = 0.1              # Parametric modulation amplitude for mode 1
h2 = 0.1              # Parametric modulation amplitude for mode 2

# Base resonance frequency (Hz)
base_freq = 10e6      # 10 MHz

# Define ODE system for the nanomechanical resonator
@test_jax.jit
def nanoresonator(state, t, omega_d, omega_p):
    x1, v1, x2, v2 = state
    
    # Parametric modulation of stiffness
    k1_t = k1 * (1 + h1 * jnp.sin(omega_p * t))
    k2_t = k2 * (1 + h2 * jnp.sin(omega_p * t))
    
    # Mode 1 dynamics
    dx1_dt = v1
    dv1_dt = (
        F1 * jnp.cos(omega_d * t)
        - c1 * v1 - gamma1 * v1**3
        - k1_t * x1 - alpha1 * x1**3 - beta1 * x1**5
        - kc * (x1 - x2) - delta * (x1 - x2)**3
        - 0.01 * x1 * v2**2  # Cross-mode damping
    ) / m1
    
    # Mode 2 dynamics
    dx2_dt = v2
    dv2_dt = (
        F2 * jnp.cos(omega_d * t)
        - c2 * v2 - gamma2 * v2**3
        - k2_t * x2 - alpha2 * x2**3 - beta2 * x2**5
        - kc * (x2 - x1) - delta * (x2 - x1)**3
        - 0.01 * x2 * v1**2  # Cross-mode damping
    ) / m2
    
    return jnp.array([dx1_dt, dv1_dt, dx2_dt, dv2_dt])

# Simulation function for a given driving frequency
def simulate_response(omega):
    # Set the driving and parametric frequencies
    omega_d = omega
    omega_p = 2 * omega
    
    # Initial state: [x1, v1, x2, v2]
    y0 = jnp.array([0.0, 0.0, 0.0, 0.0])
    
    # Time settings - adjusted for MHz frequencies
    T = 2 * jnp.pi / omega        # period in seconds
    num_periods = 200             # number of periods to simulate
    T_end = num_periods * T       # total simulation time
    num_points = 2000             # number of time points
    t = jnp.linspace(0, T_end, num_points)
    
    # Integrate the ODE system - passing only scalars and arrays
    sol = odeint(nanoresonator, y0, t, omega_d, omega_p)
    
    # Extract displacements
    x1 = sol[:, 0]
    x2 = sol[:, 2]
    
    # Use only the last 20% of the simulation for steady state
    idx_cut = int(0.8 * t.shape[0])
    x1_steady = x1[idx_cut:]
    x2_steady = x2[idx_cut:]
    
    # Calculate amplitudes
    amp1 = (jnp.max(x1_steady) - jnp.min(x1_steady)) / 2
    amp2 = (jnp.max(x2_steady) - jnp.min(x2_steady)) / 2
    
    return amp1, amp2

# JIT compile the simulation function
simulate_response_jit = test_jax.jit(simulate_response)

# Time the frequency sweep with multiple runs
num_runs = 3
execution_times = []

for i in range(num_runs):
    start_time = time.time()
    
    # Frequency sweep from near base frequency to 70 MHz
    omega_values = jnp.linspace(base_freq * 0.9, 70e6, 100)
    
    # Use regular loop since our function returns two values
    amplitudes_mode1 = []
    amplitudes_mode2 = []
    for omega in omega_values:
        amp1, amp2 = simulate_response_jit(omega)
        amplitudes_mode1.append(float(amp1))
        amplitudes_mode2.append(float(amp2))
    
    end_time = time.time()
    run_time = end_time - start_time
    execution_times.append(run_time)
    print(f"Run {i+1}/{num_runs}: {run_time:.4f} seconds")

# Calculate statistics
avg_time = np.mean(execution_times)
min_time = np.min(execution_times)
max_time = np.max(execution_times)
std_time = np.std(execution_times)

print(f"\nTiming statistics over {num_runs} runs:")
print(f"Average execution time: {avg_time:.4f} seconds")
print(f"Min: {min_time:.4f}s, Max: {max_time:.4f}s, Std Dev: {std_time:.4f}s")

# Convert to NumPy arrays for plotting
omega_values_np = np.array(omega_values)
amplitudes_mode1_np = np.array(amplitudes_mode1)
amplitudes_mode2_np = np.array(amplitudes_mode2)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(omega_values_np/1e6, amplitudes_mode1_np, 'b.-', label='Mode 1')
plt.plot(omega_values_np/1e6, amplitudes_mode2_np, 'r.-', label='Mode 2')
plt.xlabel('Driving Frequency (MHz)')
plt.ylabel('Steady-State Amplitude (m)')
plt.title('Frequency Response of Nanomechanical Resonator (JAX)')
plt.legend()
plt.grid(True)

# Add a second x-axis at the top showing normalized frequency
ax1 = plt.gca()
ax2 = ax1.twiny()
ax2.set_xlabel('Normalized Frequency (ω/ω₀)')
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(np.linspace(ax1.get_xlim()[0], ax1.get_xlim()[1], 5))
ax2.set_xticklabels([f'{x:.2f}' for x in np.linspace(float(omega_values[0]/base_freq), float(omega_values[-1]/base_freq), 5)])

plt.tight_layout()
plt.show()
