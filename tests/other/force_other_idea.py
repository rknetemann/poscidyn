import numpy as np

def f(Q, omega_0, omega, C):
    return np.sqrt(((omega_0**2 * C**2 / Q**2) - omega**2)**2 + (omega * omega_0 / Q)**2)

Q = 10.0
omega_0 = 1.0
C_values = np.linspace(1.0, 100.0, 1000)
omega_values = np.linspace(0.1, 4.0, 1000)

F_values = np.array([[f(Q, omega_0, omega, C) for C in C_values] for omega in omega_values])

import matplotlib.pyplot as plt
C_grid, omega_grid = np.meshgrid(C_values, omega_values)
plt.figure(figsize=(10, 6))
contour = plt.contourf(C_grid, omega_grid, F_values, levels=50, cmap='viridis')
plt.colorbar(contour, label='Force Magnitude F')
plt.xlabel('C')
plt.ylabel('Driving Frequency ω')       
plt.title('Force Magnitude F as a function of C and ω')
plt.show()