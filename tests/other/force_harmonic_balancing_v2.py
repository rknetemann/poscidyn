import numpy as np
import matplotlib.pyplot as plt


def f(Q, omega_0, gamma, eta):
    return (omega_0**2 / Q) * np.sqrt(4 * eta / (3 * gamma)) * np.sqrt(1 + eta)


def fwhm(omega_0, Q):
    return omega_0 / (2 * Q)


omega_0 = 2.95
gamma = 1.25e-3
eta_values = np.linspace(0.01, 1.0, num=30)
colormap = plt.cm.viridis

Q_PER_DECADE = 30  # adjustable density for log-spaced Q samples
exp_min, exp_max = 0, 5
num_q = Q_PER_DECADE * (exp_max - exp_min) + 1
Q_values = np.logspace(exp_min, exp_max, num=num_q)

# Figure 1: Q sweep (eta fixed at 1.0)
fig1, ax1 = plt.subplots(figsize=(10, 6))
eta_fixed = 1.0
q_norm = plt.Normalize(Q_values.min(), Q_values.max())
responses_q = f(Q_values, omega_0, gamma, eta_fixed)

print("[Q sweep] Summary statistics (eta fixed at 1.00)")
print(
    f"Overall -> min={responses_q.min():.4f}, "
    f"max={responses_q.max():.4f}, avg={responses_q.mean():.4f}"
)

ax1.plot(
    Q_values,
    responses_q,
    color=colormap(0.3),
    linewidth=1.8,
)
ax1.set_xscale("log")
ax1.set_xlabel("Q")
ax1.set_ylabel("f")
ax1.set_title(r"Force required to activate nonlinear term ($\eta$ = {:.2f})".format(eta_fixed))
ax1.grid(True, alpha=0.3, which="both", linestyle="--", linewidth=0.5)
sm_q = plt.cm.ScalarMappable(cmap=colormap, norm=q_norm)
sm_q.set_array([])
cbar_q = fig1.colorbar(sm_q, ax=ax1, pad=0.02)
cbar_q.set_label("Q value", rotation=270, labelpad=15)

# Figure 2: eta sweep at fixed Q
fixed_Q = 10.0
fig2, ax2 = plt.subplots(figsize=(10, 6))
responses_eta = f(fixed_Q, omega_0, gamma, eta_values)

print(f"[eta sweep] Summary statistics (Q fixed at {fixed_Q:.0f})")
print(
    f"Overall -> min={responses_eta.min():.4f}, "
    f"max={responses_eta.max():.4f}, avg={responses_eta.mean():.4f}"
)

ax2.plot(
    eta_values,
    responses_eta,
    color=colormap(0.6),
    linewidth=2.0,
)
ax2.set_xlabel(r"$\eta$")
ax2.set_ylabel("f")
ax2.set_title(f"Force required to activate nonlinear term at Q={fixed_Q:.0f} across eta values")
ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

plt.show()
