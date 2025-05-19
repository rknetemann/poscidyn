import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

# -------------------------------
# Simulation Parameters
# -------------------------------
R = 1.0  # Normalized drum radius

# Define a list of (m, n) mode pairs.
mode_list = [
    (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2),
    (2, 1), (2, 2),
    (3, 1)
]


# -------------------------------
# Create a Polar Grid and Convert to Cartesian
# -------------------------------
Nr = 200
Ntheta = 200
r = np.linspace(0, R, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta)
R_mesh, Theta_mesh = np.meshgrid(r, theta)

X = R_mesh * np.cos(Theta_mesh)
Y = R_mesh * np.sin(Theta_mesh)

# -------------------------------
# Compute the Modes
# -------------------------------
modes = []
for (m, n) in mode_list:
    bessel_zero = jn_zeros(m, n)[-1]
    phi = jn(m, bessel_zero * R_mesh)
    if m > 0:
        phi *= np.cos(m * Theta_mesh)
    amp_max = np.max(np.abs(phi))
    modes.append({'m': m, 'n': n, 'phi': phi, 'amp_max': amp_max})

# -------------------------------
# Set Up the Figure for 3D Visualization
# -------------------------------
num_modes = len(modes)
cols = 4
rows = int(np.ceil(num_modes / cols))

fig, axs = plt.subplots(rows, cols, subplot_kw={'projection': '3d'},
                        figsize=(4 * cols, 3 * rows))
axs = axs.flatten()

for ax, mode in zip(axs, modes):
    m, n = mode['m'], mode['n']
    phi = mode['phi']
    amp_max = mode['amp_max']

    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-amp_max, amp_max)
    ax.plot_surface(X, Y, phi, cmap='viridis', edgecolor='none')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax._axis3don = False

    # Add m and n as text under the plot
    ax.text2D(0.5, -0.12, f"m={m}, n={n}", transform=ax.transAxes,
              ha='center', fontsize=9)

# Hide unused subplots
for i in range(len(modes), len(axs)):
    axs[i].axis('off')

plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.show()
