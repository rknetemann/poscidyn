import numpy as np
import matplotlib.pyplot as plt

# Duffing parameters
w0, Q, gamma, fdrive, omega = 1.0, 20.0, 1.0, 1.0, 1.4

# Period and mesh
T = 2*np.pi/omega
m = 6
tau_nodes = np.linspace(0, 1, m+1)
h = np.diff(tau_nodes)

# Gauss–Legendre 2-point nodes on [0,1]
c = np.array([0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6])
xi_nodes = np.concatenate(([0.0], c, [1.0]))

# Linear (harmonic) approximate solution
def linear_guess(t):
    den = (w0**2 - omega**2)**2 + (w0*omega/Q)**2
    A = fdrive / np.sqrt(den)
    phi = np.arctan2(w0*omega/Q, w0**2 - omega**2)
    x = A*np.cos(omega*t - phi)
    return x

# --- Figure 1: Gauss collocation layout ---
fig, ax = plt.subplots(figsize=(7,2))
for k in range(m):
    tau_L, tau_R = tau_nodes[k], tau_nodes[k+1]
    ax.plot([tau_L, tau_R], [0,0], color="gray", lw=2)
    ax.plot([tau_L, tau_R], [0,0], 'k-', lw=2)
    # Collocation points
    tau_c = tau_L + h[k]*c
    ax.plot(tau_c, np.zeros_like(tau_c), 'ro')
ax.plot(tau_nodes, np.zeros_like(tau_nodes), 'ko', label="element nodes")
ax.set_yticks([])
ax.set_xlabel(r"Normalized time $\tau \in [0,1]$")
ax.set_title("Gauss collocation points within each time element")
ax.legend()
plt.tight_layout()

# --- Figure 2: Lagrange basis functions (3-node polynomial) ---
xi = np.linspace(0,1,100)
nodes = np.array([0.0, *c, 1.0])
def lagrange_basis(nodes, x):
    L = np.ones((len(x), len(nodes)))
    for j in range(len(nodes)):
        for k in range(len(nodes)):
            if k==j: continue
            L[:,j] *= (x - nodes[k])/(nodes[j]-nodes[k])
    return L

L = lagrange_basis(nodes, xi)
plt.figure(figsize=(6,3))
for j in range(L.shape[1]):
    plt.plot(xi, L[:,j], label=fr"$L_{j}(\xi)$")
plt.plot(nodes, np.eye(len(nodes)), 'ko')
plt.xlabel(r"Local coordinate $\xi$")
plt.ylabel("Basis function value")
plt.title("Lagrange shape functions in one element")
plt.legend()
plt.tight_layout()

# --- Figure 3: Approximate periodic orbit built from elements ---
# Build piecewise interpolation from "true" response
t_fine = np.linspace(0, T, 1000)
x_true = linear_guess(t_fine)

# Fake "collocation solution" as smooth stitched linear response
taus_plot = np.linspace(0,1,500)
x_periodic = linear_guess(T*taus_plot)

plt.figure(figsize=(7,3))
plt.plot(T*taus_plot, x_periodic, label="collocation approximation", lw=2)
plt.scatter([0,T], [x_periodic[0], x_periodic[-1]], color="r", zorder=5)
plt.text(T*0.5, 0.9*np.max(x_periodic), "periodic boundary:\n$x(0)=x(T)$", ha="center")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Periodic Duffing solution (conceptually via collocation)")
plt.tight_layout()

# --- Figure 4: Concept of residuals at collocation points ---
plt.figure(figsize=(7,3))
tt = np.linspace(0, T, 300)
x = linear_guess(tt)
xdot = np.gradient(x, tt)
x2dot = np.gradient(xdot, tt)
R = x2dot + (w0/Q)*xdot + w0**2*x + gamma*x**3 - fdrive*np.cos(omega*tt)
plt.plot(tt/T, R, lw=2)
plt.xlabel(r"Normalized time $\tau = t/T$")
plt.ylabel(r"Residual $R(\tau)$")
plt.title("Collocation residuals → forced to zero at Gauss points")
plt.axhline(0, color='k', lw=1)
plt.tight_layout()

plt.show()
