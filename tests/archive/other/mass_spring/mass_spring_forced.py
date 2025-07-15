import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants for the mass-spring system
m = 1.0       # mass (kg)
c = 0.2       # damping coefficient (kg/s)
k = 2.0       # spring stiffness (N/m)
A = 1.0       # amplitude of the external force (N)
omega = 0.5   # angular frequency of the external force (rad/s)
omega = np.sqrt(k/m)  # natural frequency of the system
omega -= 0.6  # adjust to avoid resonance

c_critical = 2*np.sqrt(k*m)  # critical damping coefficient
damping_ratio = c / c_critical  # damping ratio
quality_factor = np.sqrt(1-damping_ratio**2) / (2 * damping_ratio)  # quality factor
print("Damping ratio:", damping_ratio)
print("Quality factor:", quality_factor)

# Add a mode parameter to switch between "forced" and "kick" modes
mode = "forced"  # Options: "forced", "kick"

# Define the ODE system for the mass-spring system:
# m * x'' + c * x' + k * x = A * cos(omega * t) (forced mode)
# m * x'' + c * x' + k * x = 0 (kick mode)
def ode_system(t, y):
    x, v = y
    dxdt = v
    if mode == "forced":
        dvdt = (A * np.cos(omega * t) - c * v - k * x) / m
    elif mode == "kick":
        dvdt = (-c * v - k * x) / m
    else:
        raise ValueError("Invalid mode. Choose 'forced' or 'kick'.")
    return [dxdt, dvdt]

# Simulation parameters
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Adjust initial_state for "kick" mode to include an initial velocity
if mode == "kick":
    initial_state = [0.0, 1.0]  # starting with x=0 and an initial velocity
else:
    initial_state = [1.0, 0.0]  # starting with x=1 and zero velocity

# Solve the ODE
sol = solve_ivp(ode_system, t_span, initial_state, t_eval=t_eval)
x_t = sol.y[0]

# Calculate potential and kinetic energy
potential_energy = 0.5 * k * x_t**2
kinetic_energy = 0.5 * m * sol.y[1]**2  # sol.y[1] is the velocity

# Set up the figure with two subplots:
# Left subplot: mass-spring animation
# Right subplot: x vs time plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: mass-spring system animation
# Draw a fixed wall at x = 0 and the spring connecting to the mass at x position.
wall_x = [0, 0]
wall_y = [0.5, 1.5]
ax1.plot(wall_x, wall_y, 'k-', lw=4)  # represent the wall

# Initialize the spring line and the mass as a red circle
spring_line, = ax1.plot([], [], 'b-', lw=2, label='Spring')
mass_point, = ax1.plot([], [], 'ro', markersize=12, label='Mass')

# Set limits for the animation (assume mass moves horizontally)
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(0, 2)
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y')
ax1.set_title('Mass-Spring System')
ax1.legend()
ax1.grid(True)

# Right subplot: x vs time plot with energy lines
line, = ax2.plot([], [], 'b-', lw=2, label='x(t)')
potential_line, = ax2.plot([], [], 'g--', lw=1.5, label='Potential Energy')
kinetic_line, = ax2.plot([], [], 'r--', lw=1.5, label='Kinetic Energy')
current_point, = ax2.plot([], [], 'ro')  # current time position marker
ax2.set_xlim(t_span[0], t_span[1])
ax2.set_ylim(min(x_t) - 0.5, max(x_t) + 0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('x (m) / Energy (J)')
ax2.set_title('Displacement x(t) and Energy vs Time')
ax2.legend()
ax2.grid(True)

# Initialization function for the animation.
def init():
    spring_line.set_data([], [])
    mass_point.set_data([], [])
    line.set_data([], [])
    potential_line.set_data([], [])
    kinetic_line.set_data([], [])
    current_point.set_data([], [])
    return spring_line, mass_point, line, potential_line, kinetic_line, current_point

# Update function for the animation.
def update(frame):
    # Update mass-spring visualization on left subplot.
    # For simplicity, assume the spring starts at the wall (x=0) and ends at the mass at x = x_t[frame]
    mass_x = x_t[frame]
    # Create a spring shape using a sine curve for a better visual effect.
    spring_x = np.linspace(0, mass_x, 100)
    spring_y = 0.1 * np.sin(10 * np.pi * spring_x / (mass_x if mass_x != 0 else 1)) + 1.0
    spring_line.set_data(spring_x, spring_y)
    mass_point.set_data([mass_x], [1.0])  # mass is drawn at y=1 for simplicity

    # Update the x vs time graph and energy lines on right subplot.
    t_current = t_eval[:frame+1]
    x_current = x_t[:frame+1]
    potential_current = potential_energy[:frame+1]
    kinetic_current = kinetic_energy[:frame+1]
    line.set_data(t_current, x_current)
    potential_line.set_data(t_current, potential_current)
    kinetic_line.set_data(t_current, kinetic_current)
    current_point.set_data([t_eval[frame]], [x_t[frame]])
    
    return spring_line, mass_point, line, potential_line, kinetic_line, current_point

# Create the animation object.
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=25)

print("Mode:", mode)
print("Max amplitude:", np.max(np.abs(x_t)))

plt.tight_layout()
plt.show()
