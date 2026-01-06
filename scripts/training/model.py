import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree

class MultiLayerPerceptron(eqx.Module):
    layers: list

    def __init__(self, x_shape, y_shape, key):
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        self.layers = [
            eqx.nn.Linear(x_shape, 64, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(64, 64, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(64, 64, key=k5),
            jax.nn.relu,
            eqx.nn.Linear(64, 32, key=k6),
            jax.nn.relu,
            eqx.nn.Linear(32, y_shape, key=k7),
        ]

    def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        for layer in self.layers:
            x = layer(x)
        return x