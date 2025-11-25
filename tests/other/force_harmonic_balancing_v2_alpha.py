import numpy as np
import matplotlib.pyplot as plt

Q = 10.0
omega_0 = 1.0
omega = np.linspace(0.1, 3.0, num=300)
eta_values = np.linspace(0.01, 1.0, num=30)
colormap = plt.cm.viridis

alpha = 0.0267

def A_nonlinear(omega_0_1, omega_0_2, alpha, eta_1, eta_2):
    return omega_0_1**2 * omega_0_2**2 / alpha * np.sqrt(eta_1 * eta_2)

def f(A, Q, omega_0, omega):
    return A * np.sqrt((omega_0**2 - omega**2)**2 + (omega_0 * omega / Q)**2)

plt.figure(figsize=(8, 5))
norm = plt.Normalize(vmin=eta_values.min(), vmax=eta_values.max())

for eta, color in zip(eta_values, colormap(norm(eta_values))):
    amplitude = A_nonlinear(omega_0, omega_0, alpha, eta, eta)
    response = f(amplitude, Q, omega_0, omega)
    plt.plot(omega, response, color=color, lw=1.5)

sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

plt.xlabel("omega")
plt.ylabel("f(omega)")
plt.title("Force amplitude vs angular frequency for varying eta")
plt.tight_layout()
plt.show()


f_nonlinear = 1.0

def A_nonlinear(f_nonlinear, Q, omega_0, omega):
    return f_nonlinear / np.sqrt((omega_0**2 - omega**2)**2 + (omega_0 * omega / Q)**2)

def alpha_nonlinear(omega_0_1, omega_0_2, A, eta_1, eta_2):
    return omega_0_1**2 * omega_0_2**2 / (A * np.sqrt(eta_1 * eta_2))

plt.figure(figsize=(8, 5))
norm = plt.Normalize(vmin=eta_values.min(), vmax=eta_values.max())  
for eta, color in zip(eta_values, colormap(norm(eta_values))):
    amplitude = A_nonlinear(f_nonlinear, Q, omega_0, omega)
    response = alpha_nonlinear(omega_0, omega_0, amplitude, eta, eta)
    plt.plot(omega, response, color=color, lw=1.5)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
plt.xlabel("omega")
plt.ylabel("alpha(omega)")
plt.title("Nonlinear coefficient vs angular frequency for varying eta")
plt.tight_layout()
plt.show()