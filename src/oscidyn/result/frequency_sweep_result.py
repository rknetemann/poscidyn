from jax import tree_util

from ..model.abstract_model import AbstractModel
from ..excitation.abstract_excitation import AbstractExcitation

class FrequencySweepResult:
    def __init__(self, model: AbstractModel, excitor: AbstractExcitation, periodic_solutions, sweeped_periodic_solutions):
        self.model: AbstractModel = model
        self.excitor: AbstractExcitation = excitor

        self.periodic_solutions = periodic_solutions
        self.sweeped_periodic_solutions = sweeped_periodic_solutions

# Register FrequencySweepResult as a pytree
def _tree_flatten(obj):
    leaves = (obj.model, obj.excitor, obj.periodic_solutions, obj.sweeped_periodic_solutions)
    return leaves, None

def _tree_unflatten(aux_data, leaves):
    model, excitor, periodic_solutions, sweeped_periodic_solutions = leaves
    return FrequencySweepResult(model, excitor, periodic_solutions, sweeped_periodic_solutions)

tree_util.register_pytree_node(
    FrequencySweepResult, _tree_flatten, _tree_unflatten
)
