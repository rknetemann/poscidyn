import numpy as np
import matplotlib.pyplot as plt

# System parameters
m = 1.0             
k = 1.0             
omega0 = np.sqrt(k/m)
Q = 100.0           
c = m * omega0 / Q  
F = 1.2             # increased drive amplitude for bistability
alpha_soft = -10.0  

# Frequency range
omega = np.linspace(0, 10.5*omega0, 1200)

# Arrays for branches
low = np.full_like(omega, np.nan)
mid = np.full_like(omega, np.nan)
high = np.full_like(omega, np.nan)

# Track where three real roots exist
has_three = np.zeros_like(omega, dtype=bool)

# Solve cubic for each frequency
for i, w in enumerate(omega):
    A = k - m*w**2
    B = 0.75 * alpha_soft
    D = c * w
    coeffs = [B**2, 2*A*B, A**2 + D**2, -F**2]
    roots = np.roots(coeffs)
    y_roots = [r.real for r in roots if np.isclose(r.imag, 0, atol=1e-6) and r.real > 0]
    if len(y_roots) == 3:
        has_three[i] = True
    x_roots = np.sqrt(y_roots)
    x_roots.sort()
    if len(x_roots) == 1:
        low[i] = x_roots[0]
    elif len(x_roots) == 3:
        low[i], mid[i], high[i] = x_roots

# Find bifurcation frequencies
indices = np.where(has_three)[0]
omega_bif = [omega[indices[0]], omega[indices[-1]]] if indices.size >= 2 else []

# Plotting
plt.figure(figsize=(8,5))
plt.plot(omega/omega0, low, label='Lower (stable)')
plt.plot(omega/omega0, high, label='Upper (stable)')
plt.plot(omega/omega0, mid, '--', label='Middle (unstable)')
for wb in omega_bif:
    plt.axvline(wb/omega0, color='k', linestyle=':', linewidth=1)
    plt.text(wb/omega0, max(high)*0.9, f'ω={wb/omega0:.3f}ω₀', 
             rotation=90, va='center', ha='right', fontsize=8)
plt.title("Duffing Oscillator Response with Bifurcation Points (Softening α = -10, F=0.2)")
plt.xlabel("Normalized Frequency ω/ω₀")
plt.ylabel("Amplitude |X|")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
