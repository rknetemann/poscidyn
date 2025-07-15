# results.py
import matplotlib.pyplot as plt

from oscidyn.models import AbstractModel
from oscidyn.solver import AbstractSolver
from oscidyn.constants import SweepDirection
import oscidyn.constants as const

class FrequencySweepResult():
    def __init__(self, model: AbstractModel, sweep_direction: SweepDirection,
                 driving_frequencies, driving_amplitudes,
                 steady_state_displacement_amplitude, steady_state_velocity_amplitude,
                 total_steady_state_displacement_amplitude, total_steady_state_velocity_amplitude,
                 solver: AbstractSolver):
        
        self.driving_frequencies = driving_frequencies # Shape: (n_driving_frequencies,)
        self.driving_amplitudes = driving_amplitudes # Shape: (n_driving_amplitudes,)
        self.steady_state_displacement_amplitude = steady_state_displacement_amplitude # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        self.steady_state_velocity_amplitude = steady_state_velocity_amplitude # Shape: (n_driving_frequencies * n_driving_amplitudes, n_modes)
        self.total_steady_state_displacement_amplitude = total_steady_state_displacement_amplitude # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        self.total_steady_state_velocity_amplitude = total_steady_state_velocity_amplitude # Shape: (n_driving_frequencies * n_driving_amplitudes,)
        self.sweep_direction = sweep_direction # SweepDirection Enum

    def plot(self):
        plt.figure()
        plt.plot(self.driving_frequencies, self.total_steady_state_displacement_amplitude, label='Displacement Amplitude')
        plt.plot(self.driving_frequencies, self.total_steady_state_velocity_amplitude, label='Velocity Amplitude')
        plt.xlabel('Driving Frequency')
        plt.ylabel('Amplitude')
        plt.title('Frequency Sweep Results')
        plt.legend()
        plt.grid(True)
        plt.show()