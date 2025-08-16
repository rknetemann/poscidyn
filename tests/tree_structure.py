from jax.tree_util import tree_structure
from oscidyn.simulation.examples.models import NonlinearOscillator

model = NonlinearOscillator.from_example(n_modes=1)
print("Model structure:")
print(tree_structure(model))