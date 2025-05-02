import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# 1) Randomize coefficients
np.random.seed(42)

# Linear frequencies (rad/s)
omega1 = 2*np.pi * np.random.uniform(0.5e6, 2.0e6)   # between 0.5 and 2 MHz
omega2 = 2*np.pi * np.random.uniform(0.5e6, 2.0e6)

# 2nd-order coefficients α_{ij}^(1) for eqn of x:
alpha11_1 = np.random.uniform(-1e3, 1e3)
alpha12_1 = np.random.uniform(-1e3, 1e3)
alpha22_1 = np.random.uniform(-1e3, 1e3)

# 3rd-order coefficients γ_{ijk}^(1)
gamma111_1 = np.random.uniform(-1e-3, 1e-3)
gamma112_1 = np.random.uniform(-1e-3, 1e-3)
gamma122_1 = np.random.uniform(-1e-3, 1e-3)
gamma222_1 = np.random.uniform(-1e-3, 1e-3)

# and similarly for the q–equation:
alpha11_2 = np.random.uniform(-1e3, 1e3)
alpha12_2 = np.random.uniform(-1e3, 1e3)
alpha22_2 = np.random.uniform(-1e3, 1e3)
gamma111_2 = np.random.uniform(-1e-3, 1e-3)
gamma112_2 = np.random.uniform(-1e-3, 1e-3)
gamma122_2 = np.random.uniform(-1e-3, 1e-3)
gamma222_2 = np.random.uniform(-1e-3, 1e-3)

# Forcing amplitude
F0 = 1.0

print("Randomized coefficients:")
print(f"  ω1 = {omega1:.3e} rad/s,   ω2 = {omega2:.3e} rad/s")
print("  α^(1):", alpha11_1, alpha12_1, alpha22_1)
print("  γ^(1):", gamma111_1, gamma112_1, gamma122_1, gamma222_1)
print("  α^(2):", alpha11_2, alpha12_2, alpha22_2)
print("  γ^(2):", gamma111_2, gamma112_2, gamma122_2, gamma222_2)
print(f"  Forcing F0 = {F0}\n")

# Define the ODE system with forcing on x
def odes(t, y, Omega):
    x, xd, q, qd = y
    # nonlinear forces in x‐eqn
    nonlin1 = (alpha11_1*x**2 + alpha12_1*x*q + alpha22_1*q**2
               + gamma111_1*x**3 + gamma112_1*x**2*q
               + gamma122_1*x*q**2 + gamma222_1*q**3)
    # nonlinear forces in q‐eqn
    nonlin2 = (alpha11_2*x**2 + alpha12_2*x*q + alpha22_2*q**2
               + gamma111_2*x**3 + gamma112_2*x**2*q
               + gamma122_2*x*q**2 + gamma222_2*q**3)
    # equations
    ddx = -omega1**2 * x - nonlin1 + F0*np.cos(Omega*t)
    ddq = -omega2**2 * q - nonlin2
    return [xd, ddx, qd, ddq]

# Frequency sweep parameters
f_min, f_max, Nf = 1e3, 100e6, 300
freqs = np.linspace(f_min, f_max, Nf)
amps = np.zeros_like(freqs)

for i, f in enumerate(freqs):
    Omega = 2*np.pi*f
    # Integrate for N_cycles to let transients die out
    N_cycles = 200
    t_end = N_cycles * 2*np.pi / Omega
    t_eval = np.linspace(0, t_end, 5000)
    y0 = [0.0, 0.0, 0.0, 0.0]
    sol = solve_ivp(odes, [0, t_end], y0, t_eval=t_eval, args=(Omega,),
                    atol=1e-6, rtol=1e-6)
    x_t = sol.y[0]
    # take last 10% of the time series to measure amplitude
    tail = x_t[int(0.9*len(x_t)):]
    amps[i] = 0.5*(np.max(tail) - np.min(tail))

# Plot the steady‑state amplitude response
plt.figure(figsize=(8,4))
plt.semilogx(freqs, amps, lw=1.5)
plt.xlabel('Drive frequency f (Hz)')
plt.ylabel('Steady‑state amplitude of x')
plt.title('Nonlinear Coupled‑Oscillator Frequency Response')
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
