import numpy as np

Q = np.array([10.0, 20.0])
omega_0 = np.array([1.0, 2.0])
gamma = np.array([1.0e-3, 1.0e-3])
alpha = 1.0e-4

eta = np.array([1.0, 1.0])

omega = np.linspace(0.1, 3.0, 200)
omega = 1.0

t = np.linspace(0, 400, 1000)

Ax = 4/3 * (eta[0]*omega_0[0]**2 / gamma[0])
Ay = 4/3 * (eta[1]*omega_0[1]**2 / gamma[1])

x = Ax * np.cos(omega * t)
x_dot = -omega * Ax * np.sin(omega * t)
x_ddot = -omega**2 * Ax * np.cos(omega * t)

y = Ay * np.cos(omega * t)
y_dot = -omega * Ay * np.sin(omega * t)
y_ddot = -omega**2 * Ay * np.cos(omega * t)

fx = (x_ddot + (omega_0[0] / Q[0]) * x_dot + omega_0[0]**2 * x + gamma[0] * x**3) / np.cos(omega * t)
fy = (y_ddot + (omega_0[1] / Q[1]) * y_dot + omega_0[1]**2 * y + gamma[1] * y**3) / np.cos(omega * t)

print(np.max(fx))
print(np.max(fy))