import oscidyn
import numpy as np

# time, displacement, velocity = oscidyn.time_response(
#     model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
#     driving_frequency = 1.0,
#     driving_amplitude = 0.5,
#     initial_displacement = np.zeros((1,)),
#     initial_velocity = np.zeros((1,)),
#     solver = oscidyn.StandardSolver(t_end=1000.0, n_steps=5000, max_steps=20192),
# )

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(time, displacement.squeeze(), label='Displacement')
# plt.plot(time, velocity.squeeze(), label='Velocity')
# plt.xlabel('Time')
# plt.ylabel('Response')
# plt.title('Time Response of the Nonlinear Oscillator')
# plt.legend()
# plt.grid(True)
# plt.show()

# frequency_sweep = oscidyn.frequency_sweep(
#     model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
#     sweep_direction = oscidyn.SweepDirection.FORWARD,
#     driving_frequencies = np.linspace(0.1, 4, 300),
#     driving_amplitudes = np.linspace(0.1, 1.5, 100),
#     solver = oscidyn.StandardSolver(t_end=1000.0, n_steps=5000, max_steps=200192),
# )

frequency_sweep = oscidyn.frequency_sweep(
    model = oscidyn.NonlinearOscillator.from_example(n_modes=1),
    sweep_direction = oscidyn.SweepDirection.FORWARD,
    driving_frequencies = np.linspace(0.1, 4, 300),
    driving_amplitudes = np.linspace(0.1, 1.5, 100),
    solver = oscidyn.SteadyStateSolver()
)

