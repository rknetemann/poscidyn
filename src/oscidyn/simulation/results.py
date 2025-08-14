# results.py
import matplotlib.pyplot as plt
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from oscidyn.simulation.models import AbstractModel
from oscidyn.simulation.solver import AbstractSolver
from oscidyn.simulation.constants import SweepDirection
import oscidyn.simulation.constants as const

@jax.tree_util.register_pytree_node_class
@dataclass
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

    # PyTree protocol
    def tree_flatten(self):
        children = (
            self.driving_frequencies,
            self.driving_amplitudes,
            self.steady_state_displacement_amplitude,
            self.steady_state_velocity_amplitude,
            self.total_steady_state_displacement_amplitude,
            self.total_steady_state_velocity_amplitude,
        )
        aux = {
            "model": self.model,
            "sweep_direction": self.sweep_direction,
            "solver": self.solver,
        }
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            driving_frequencies,
            driving_amplitudes,
            steady_state_displacement_amplitude,
            steady_state_velocity_amplitude,
            total_steady_state_displacement_amplitude,
            total_steady_state_velocity_amplitude,
        ) = children
        return cls(
            model=aux["model"],
            sweep_direction=aux["sweep_direction"],
            driving_frequencies=driving_frequencies,
            driving_amplitudes=driving_amplitudes,
            steady_state_displacement_amplitude=steady_state_displacement_amplitude,
            steady_state_velocity_amplitude=steady_state_velocity_amplitude,
            total_steady_state_displacement_amplitude=total_steady_state_displacement_amplitude,
            total_steady_state_velocity_amplitude=total_steady_state_velocity_amplitude,
            solver=aux["solver"],
        )

    def plot(self):
        raise NotImplementedError("Plotting not implemented for FrequencySweepResult")