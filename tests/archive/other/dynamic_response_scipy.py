import test_jax
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
def simulate_time_series_coupled(f, omega, params, dt=0.01, n_steps=8000):
    # initial state: both modes start at rest
    state = jnp.array([0.0, 0.0, 0.0, 0.0])
    t0 = 0.0

    def step_fn(carry, _):
        state, t = carry
        new_state = rk4_step_coupled(state, t, dt, f, omega, params)
        new_t = t + dt
        # record only the displacement of the primary mode x1
        return (new_state, new_t), new_state[0]
    
    (final_state, final_t), x1_series = test_jax.lax.scan(step_fn, (state, t0), None, length=n_steps)
    return x1_series

# Compute the steady-state amplitude by discarding a fraction of the time series as transient.
def compute_amplitude_coupled(f, omega, params, dt=0.01, n_steps=8000, transient_fraction=0.5):
    x1_series = simulate_time_series_coupled(f, omega, params, dt=dt, n_steps=n_steps)
    transient_cut = int(n_steps * transient_fraction)
    steady_state = x1_series[transient_cut:]
    # Return the maximum absolute displacement in the steady state
    amp = jnp.max(jnp.abs(steady_state))
    return amp

# Main simulation parameters (dimensionless)
# Parameters for mode 1 and mode 2 (typical literature-like values)
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

# Define the resonator's base (physical) frequency.
f0 = 35e6  # 35 MHz

# Frequency sweep: dimensionless sweep from 0 to 2 corresponds to 0 to 70 MHz (physical).
frequencies = jnp.linspace(0.0, 2.0, 200)
# Convert to physical frequencies in MHz for plotting: f_phys = omega * f0
frequencies_phys = (frequencies * f0) / 1e6  # in MHz

# List of different driving force amplitudes to see the effect of nonlinearity.
drive_amplitudes = [0.1, 0.3, 0.5, 0.7]

# Settings for the time integration
dt = 0.01           # time step (dimensionless)
n_steps = 8000      # total number of time steps per frequency
transient_fraction = 0.5  # discard the first half as transient

# For each drive amplitude, compute the steady-state amplitude vs drive frequency.
results = {}
for f in drive_amplitudes:
    # Vectorize the computation over frequency.
    compute_for_freq = test_jax.jit(test_jax.vmap(
        lambda omega: compute_amplitude_coupled(f, omega, coupled_params, dt, n_steps, transient_fraction)
    ))
    amps = compute_for_freq(frequencies)
    results[f] = amps

# Convert JAX arrays to Python lists for plotting.
frequencies_phys_np = frequencies_phys.tolist()

plt.figure(figsize=(8, 5))
for f in drive_amplitudes:
    amps_np = jnp.array(results[f])
    plt.plot(frequencies_phys_np, amps_np, label=f'Drive amplitude = {f:.2f}')
plt.xlabel("Drive frequency (MHz)")
plt.ylabel("Steady-state amplitude (x₁)")
plt.title("Dynamic Frequency Response of a Nonlinear Coupled Resonator")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
