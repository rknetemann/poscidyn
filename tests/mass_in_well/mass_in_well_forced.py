import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Constants
g = 9.8       # acceleration due to gravity (m/s^2)
c = 1.5       # damping coefficient (kg/s)
m = 1.0       # mass (kg)
A = 2.0       # amplitude of the external harmonic force (N)
omega = 1.5   # angular frequency of the external harmonic force (rad/s)


# Define the ODE system (with damping and external harmonic force) for x and its velocity v (where v = dx/dt)
def ode_system(t, y):
    x, v = y
    denom = 1 + 4*x**2
    damping = (c/m) * (1+4*x**2) * v
    # External force term (applied in the x-direction)
    ext_force = (A/m) * np.cos(omega * t)
    # Modified equation: 
    # v' = [ - (4*x*v**2 + 2*g*x) + ext_force ] / (1+4*x**2) - (c/m)*v
    dvdt = (- (4*x*v**2 + 2*g*x) + ext_force) / denom - (c/m)*v
    return [v, dvdt]

# Increase the simulation time span
t_span = (0, 50)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial conditions: start at x=1 with zero velocity.
initial_state = [1.0, 0.0]
sol = solve_ivp(ode_system, t_span, initial_state, t_eval=t_eval)

# Extract x(t) and compute y(t) using the constraint y = x^2.
x_t = sol.y[0]
y_t = x_t**2

# Set up the figure with two subplots: one for the well animation, one for x vs time.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# -- Left subplot: The well with constraint y = x^2 --
x_range = np.linspace(-2, 2, 400)
ax1.plot(x_range, x_range**2, 'k-', label='Constraint: y = x^2')
point, = ax1.plot([], [], 'ro', markersize=8, label='Mass')
ax1.set_xlim(-2, 2)
ax1.set_ylim(0, 4)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Mass on the Parabolic Well')
ax1.legend()
ax1.grid(True)

# -- Right subplot: x vs time graph --
line, = ax2.plot([], [], 'b-', lw=2, label='x(t)')
current_point, = ax2.plot([], [], 'ro')  # to show current x(t) position
ax2.set_xlim(t_span[0], t_span[1])
ax2.set_ylim(min(x_t)-0.5, max(x_t)+0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('x')
ax2.set_title('Oscillatory Behavior: x(t) vs Time')
ax2.legend()
ax2.grid(True)

# Initialization function for the animation.
def init():
    point.set_data([], [])
    line.set_data([], [])
    current_point.set_data([], [])
    return point, line, current_point

# Update function for the animation.
def update(frame):
    # Update left subplot: the well animation.
    x = x_t[frame]
    y = y_t[frame]
    point.set_data([x], [y])
    
    # Update right subplot: plot x(t) up to current time.
    t_current = t_eval[:frame+1]
    x_current = x_t[:frame+1]
    line.set_data(t_current, x_current)
    # Mark the current time point with a red dot.
    current_point.set_data([t_eval[frame]], [x_t[frame]])
    
    return point, line, current_point

# Create the animation object.
ani = FuncAnimation(fig, update, frames=len(t_eval), init_func=init, blit=True, interval=25)

plt.tight_layout()
plt.show()
