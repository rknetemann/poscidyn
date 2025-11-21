import equinox as eqx
import jax
from jaxtyping import Array, Float, PyTree

class MultiLayerPerceptron(eqx.Module):
    layers: list

    def __init__(self, x_shape, y_shape, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(x_shape, 128, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 64, key=k3),
            jax.nn.relu,
            eqx.nn.Linear(64, y_shape, key=k4),
        ]

    def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "output_dim"]:
        for layer in self.layers:
            x = layer(x)
        return x