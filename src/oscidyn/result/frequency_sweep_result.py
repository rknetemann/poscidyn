from jax import tree_util

from ..model.abstract_model import AbstractModel
from ..excitation.abstract_excitation import AbstractExcitation

class FrequencySweepResult:
    def __init__(self, f_omegas, f_amps, modal_forces, Q, omega_0, alpha, gamma, periodic_solutions, sweeped_periodic_solutions):
        self.f_omegas = f_omegas
        self.f_amps = f_amps
        self.modal_forces = modal_forces
        self.Q = Q
        self.omega_0 = omega_0
        self.alpha = alpha
        self.gamma = gamma
        self.periodic_solutions = periodic_solutions
        self.sweeped_periodic_solutions = sweeped_periodic_solutions

# Register FrequencySweepResult as a pytree
def _tree_flatten(obj):
    leaves = (obj.f_omegas, obj.f_amps, obj.modal_forces,obj.Q, obj.omega_0, obj.alpha, obj.gamma, obj.periodic_solutions, obj.sweeped_periodic_solutions)
    return leaves, None

def _tree_unflatten(aux_data, leaves):
    f_omegas, f_amps, modal_forces, Q, omega_0, alpha, gamma, periodic_solutions, sweeped_periodic_solutions = leaves
    return FrequencySweepResult(f_omegas, f_amps, modal_forces, Q, omega_0, alpha, gamma, periodic_solutions, sweeped_periodic_solutions)

tree_util.register_pytree_node(
    FrequencySweepResult, _tree_flatten, _tree_unflatten
)
