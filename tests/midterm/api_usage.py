import poscidyn
import numpy as np

Q, omega_0, alpha, gamma = np.array([10.0, 20.0]), np.array([1.00, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
gamma[0,0,0,0] = 2.67e-02
gamma[1,1,1,1] = 5.40e-01
alpha[0,0,1] = 7.48e-01
alpha[1,0,0] = 3.74e-01

driving_frequency = np.linspace(0.1, 2.0, 150)
driving_amplitude = np.linspace(0.1, 1.0, 10)
modal_forces = np.array([1.0, 0.0])

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)

time_response = poscidyn.time_response(
    model = MODEL,
    driving_frequency = 1.0,
    driving_amplitude = 1.0,
    initial_displacement= np.array([0.0, 0.0]),
    initial_velocity = np.array([0.0, 0.0]),
    only_save_steady_state = True
)

