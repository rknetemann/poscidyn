import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Simulation parameters
n_masses = 101  # Number of masses (must be odd to have a unique middle node)
if n_masses % 2 == 0:
    raise ValueError("n_masses must be odd to have a unique middle node.")

m = 1.0       # Mass of each mass (kg)
k = 1000.0      # Spring constant for each spring (N/m)

# Drive parameters
drive_amplitude = 1000.0  # Amplitude of the drive force (N)
drive_frequency = 1.5  # Frequency of the drive forpce (Hz)
drive_frequency = np.sqrt(k/m)  # Natural frequency of the system
drive_frequency -= 0.0  # Adjust to avoid resonance

dt = 0.01     # Time step (s)
T = 10.0      # Total simulation time (s)
n_steps = int(T / dt)

# Initialize positions and velocities (vertical displacements)
y = np.zeros(n_masses)  # Vertical positions (initially 0)
v = np.zeros(n_masses)  # Vertical velocities (initially 0)

# Pre-allocate arrays to store simulation results for animation
y_history = np.zeros((n_steps, n_masses))
middle_history = np.zeros(n_steps)

# Determine the index of the middle mass
middle_index = n_masses // 2

# Simulation loop using Euler integration
for step in range(n_steps):
    # Save current state for animation
    y_history[step] = y.copy()
    middle_history[step] = y[middle_index]

    # Initialize acceleration array for all masses
    a = np.zeros(n_masses)
    
    # Compute acceleration for the inner masses (excluding fixed endpoints)
    for i in range(1, n_masses - 1):
        # Spring force: F = k * (y[i+1] + y[i-1] - 2*y[i])
        a[i] = k/m * (y[i+1] + y[i-1] - 2*y[i])
    
    # Apply the time-varying drive force to the middle mass
    current_time = step * dt
    drive_force = drive_amplitude * np.sin(2 * np.pi * drive_frequency * current_time)
    a[middle_index] += drive_force / m

    # Update velocities and positions for the inner masses
    v[1:-1] += a[1:-1] * dt
    y[1:-1] += v[1:-1] * dt

    # Enforce fixed boundary conditions: endpoints remain fixed
    y[0] = 0
    y[-1] = 0
    v[0] = 0
    v[-1] = 0

# Set up the figure and two subplots:
#   - ax1 for the membrane animation.
#   - ax2 for the middle mass displacement vs. time.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
x_coords = np.arange(n_masses)

# Initial plot for membrane positions
line1, = ax1.plot(x_coords, y_history[0], 'bo-', lw=2)
ax1.set_xlim(0, n_masses - 1)
ax1.set_ylim(np.min(y_history) - 0.1, np.max(y_history) + 0.1)
ax1.set_title('Membrane Positions')
ax1.set_xlabel('Mass index')
ax1.set_ylabel('Vertical displacement')

# Plot the time series for the middle mass displacement
time_array = np.linspace(0, T, n_steps)
ax2.plot(time_array, middle_history, 'r-', lw=1)
# A red dot to mark the current time point; note we wrap the scalars in lists
dot, = ax2.plot([0], [middle_history[0]], 'ro')
ax2.set_xlim(0, T)
ax2.set_ylim(np.min(middle_history) - 0.1, np.max(middle_history) + 0.1)
ax2.set_title('Middle Mass Displacement Over Time')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Vertical displacement')

def update(frame):
    # Update the membrane plot
    line1.set_ydata(y_history[frame])
    # Update the red dot on the middle mass time series plot
    current_time = frame * dt
    dot.set_data([current_time], [middle_history[frame]])
    return line1, dot

ani = FuncAnimation(fig, update, frames=n_steps, interval=20, blit=True)
plt.tight_layout()
plt.show()
