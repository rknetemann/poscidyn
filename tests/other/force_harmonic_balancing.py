import numpy as np

def f(omega, Q, omega_0, gamma, eta):
    return np.sqrt(omega_0**2/(eta*gamma)) * np.sqrt((omega_0**2 - omega**2 + 3*omega_0**2/(4*eta))**2 + (omega*omega_0/Q)**2)

def delta_omega_0(omega_0, gamma, eta):
    return 3 * gamma * omega_0**2 / (8 * omega_0 * eta * gamma)

Q = 30.0
omega_0 = 5.0
gamma = 5.00e-3 * 0.001 * 10
eta = 10.0

omega = np.linspace(0.5, 5.5, 500)
y = f(omega, Q, omega_0, gamma, eta)

min_y = np.min(y)
max_y = np.max(y)
avg_y = np.mean(y)

delta_omega = delta_omega_0(omega_0, gamma, eta)
print(f"Delta omega_0: {delta_omega}")

print(f"Min f(omega): {min_y}")
print(f"Max f(omega): {max_y}")
print(f"Avg f(omega): {avg_y}")

import matplotlib.pyplot as plt
plt.plot(omega, y)
plt.xlabel('omega')
plt.ylabel('f(omega)')
plt.title('Plot of f(omega) vs omega')
plt.grid(True)
plt.show()