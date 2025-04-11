import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn, jn_zeros

# -------------------------------
# Simulation Parameters
# -------------------------------
R = 1.0  # Normalized drum radius

# Define a list of (m, n) mode pairs.
# Here we list 12 modes; modify or extend as desired.
mode_list = [
    (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3),
    (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3)
]

# -------------------------------
# Create a Polar Grid and Convert to Cartesian
# -------------------------------
Nr = 200         # Number of radial grid points
Ntheta = 200     # Number of angular grid points

r = np.linspace(0, R, Nr)
theta = np.linspace(0, 2 * np.pi, Ntheta)
R_mesh, Theta_mesh = np.meshgrid(r, theta)

# Convert polar coordinates to Cartesian for surface plotting.
X = R_mesh * np.cos(Theta_mesh)
Y = R_mesh * np.sin(Theta_mesh)

# -------------------------------
# Compute the Modes and Their Maximum Amplitudes
# -------------------------------
modes = []
for (m, n) in mode_list:
    # Compute the nth zero for the Bessel function of order m.
    bessel_zero = jn_zeros(m, n)[-1]
    
    # Compute the spatial eigenfunction φ(r,θ).
    phi = jn(m, bessel_zero * R_mesh)
    if m > 0:
        phi *= np.cos(m * Theta_mesh)
        
    # Compute the maximum absolute amplitude.
    amp_max = np.max(np.abs(phi))
    
    modes.append({
        'm': m,
        'n': n,
        'phi': phi,
        'amp_max': amp_max
    })

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
    m = mode['m']
    n = mode['n']
    phi = mode['phi']
    amp_max = mode['amp_max']
    
    # Set plot limits to display the circular membrane in x–y and the mode amplitude in z.
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_zlim(-amp_max, amp_max)
    
    # Plot the 3D surface.
    ax.plot_surface(X, Y, phi, cmap='viridis', edgecolor='none')
    
    # Remove tick labels.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Remove grid lines.
    ax.grid(False)
    
    # Hide the entire 3D axes box (using the internal attribute).
    ax._axis3don = False
    
    # Set a title with mode info.
    ax.set_title(f"m={m}, n={n}\nMax Amp = {amp_max:.3f}", fontsize=10)

# Hide any extra subplots.
for i in range(len(modes), len(axs)):
    axs[i].axis('off')

fig.suptitle("3D Vibrational Modes of a Circular Drum", fontsize=16)

# Make the layout more compact.
plt.tight_layout()
plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0.05, right=0.95, top=0.90, bottom=0.05)
plt.show()
