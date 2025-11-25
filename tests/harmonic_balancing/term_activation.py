import numpy as np
from scipy.optimize import fsolve

def f(A, Q, omega_0, gamma):
    omega_res = np.sqrt(omega_0**2 + 3/4 * gamma * A**2)
    f = A * np.sqrt((omega_0**2 + 3/4 * gamma * A**2 - omega_res**2)**2 + (omega_0 * omega_res / Q)**2)
    return f

def activation_gamma(eta_gamma, omega_0, activation_A):
    return 4/3 * (eta_gamma * omega_0**2) / (activation_A**2)

def activation_alpha(eta_alpha, gamma, omega_0, activation_A):
    alphax = omega_0[0] * omega_0[1] / activation_A[0] * np.sqrt(eta_alpha[0] * eta_alpha[1])
    alphay = omega_0[1] * omega_0[1] / activation_A[0] * eta_alpha[0]
    return np.array([alphax, alphay])

if __name__ == "__main__":
    Q = np.array([10.0, 20.0])
    omega_0 = np.array([1.0, 3.0])
    gamma = np.array([0.0267, 0.540])

    eta_gamma = np.array([1.0, 1.0])
    eta_alpha = np.array([1.0, 1.0])

    activation_A = []

    def equation(A):
        return f(A, Q, omega_0, gamma) - 1

    activation_A = fsolve(equation, np.array([0.1, 0.1])) 
    activation_gamma_values = activation_gamma(eta_gamma, omega_0, activation_A)
    activation_alpha_values = activation_alpha(eta_alpha, gamma, omega_0, activation_A)
    print("Activation Amplitudes:", activation_A)
    print("Required Gamma for Activation:", activation_gamma_values)
    print("Required Alpha for Activation:", activation_alpha_values)