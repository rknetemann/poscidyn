from dataclasses import dataclass
from jax import tree_util
from jaxtyping import Array


@tree_util.register_pytree_node_class
@dataclass
class TimeResponse:
    time: Array
    displacement: Array
    velocity: Array

    @property
    def state(self):
        return self.displacement, self.velocity

    def __iter__(self):
        # Backwards-compatible with: ts, xs, vs = solver.time_response(...)
        return iter((self.time, self.displacement, self.velocity))

    def tree_flatten(self):
        return (self.time, self.displacement, self.velocity), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
