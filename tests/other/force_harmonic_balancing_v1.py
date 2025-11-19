import numpy as np
import matplotlib.pyplot as plt

def f(omega, Q, omega_0, gamma, eta):
    return np.sqrt(4 * eta * omega_0**2 / (3 * gamma)) * np.sqrt(
        ((1 + eta) * omega_0**2 - omega**2) ** 2 + (omega * omega_0 / Q) ** 2
    )

def fwhm(omega_0, Q):
    return omega_0 / (2 * Q)

omega_0 = 1.0
gamma = 5.00e-2
omega = np.linspace(0.5, 5.5, 500)
eta_values = np.linspace(0.01, 1.0, num=6)
colormap = plt.cm.viridis

Q_PER_DECADE = 30  # adjustable density for log-spaced Q samples
exp_min, exp_max = 0, 5
num_q = Q_PER_DECADE * (exp_max - exp_min) + 1
Q_values = np.logspace(exp_min, exp_max, num=num_q)

# Figure 1: Q sweep (eta fixed at 1.0)
fig1, ax1 = plt.subplots(figsize=(10, 6))
eta_fixed = 1.0
q_norm = plt.Normalize(Q_values.min(), Q_values.max())

for Q in Q_values:
    color = colormap(q_norm(Q))
    response = f(omega, Q, omega_0, gamma, eta_fixed)
    min_y = np.min(response)
    max_y = np.max(response)
    avg_y = np.mean(response)

    print(
        f"[Q sweep] Q={int(round(Q))}, eta={eta_fixed:.2f} -> "
        f"min={min_y:.4f}, max={max_y:.4f}, avg={avg_y:.4f}, FWHM={fwhm(omega_0, Q):.4f}"
    )

    ax1.plot(
        omega,
        response,
        color=color,
        linewidth=1.2,
    )

ax1.set_xlabel(r"$\omega$")
ax1.set_ylabel("f")
ax1.set_title(r"Force required to activate nonlinear term ($\eta$ = {:.2f})".format(eta_fixed))
ax1.grid(True, alpha=0.3)
sm_q = plt.cm.ScalarMappable(cmap=colormap, norm=q_norm)
sm_q.set_array([])
cbar_q = fig1.colorbar(sm_q, ax=ax1, pad=0.02)
cbar_q.set_label("Q value", rotation=270, labelpad=15)

# Figure 2: eta sweep at fixed Q
fixed_Q = 100.0
fig2, ax2 = plt.subplots(figsize=(10, 6))
eta_norm = plt.Normalize(eta_values.min(), eta_values.max())

for eta in eta_values:
    color = colormap(eta_norm(eta))
    response = f(omega, fixed_Q, omega_0, gamma, eta)
    min_y = np.min(response)
    max_y = np.max(response)
    avg_y = np.mean(response)

    print(
        f"[eta sweep] Q={fixed_Q:.0f}, eta={eta:.2f} -> "
        f"min={min_y:.4f}, max={max_y:.4f}, avg={avg_y:.4f}, FWHM={fwhm(omega_0, fixed_Q):.4f}"
    )

    ax2.plot(
        omega,
        response,
        color=color,
        label=f"eta={eta:.2f}",
        linewidth=1.6,
    )

ax2.set_xlabel(r"$\omega$")
ax2.set_ylabel("f")
ax2.set_title(f"Force required to activate nonlinear term at Q={fixed_Q:.0f} across eta values")
ax2.grid(True, alpha=0.3)
sm_eta = plt.cm.ScalarMappable(cmap=colormap, norm=eta_norm)
sm_eta.set_array([])
cbar_eta = fig2.colorbar(sm_eta, ax=ax2, pad=0.02)
cbar_eta.set_label("Eta value", rotation=270, labelpad=15)

plt.show()
