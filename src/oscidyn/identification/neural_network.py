import os; os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree 
import optax

INPUT_SHAPE = (3, 28, 28)
SEED = 42

class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.layers = [
            eqx.nn.Conv2d(3, 32, kernel_size=3, key=key1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),

            eqx.nn.Conv2d(32, 64, kernel_size=3, key=key2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2),

            eqx.nn.Conv2d(64, 128, kernel_size=3, key=key3),
            jax.nn.relu,
            
            jnp.ravel,

            eqx.nn.Linear(51200, 512, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(512, 64, key=key5),
            jax.nn.relu,
            eqx.nn.Linear(64, 2, key=key6),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "{INPUT_SHAPE[0]} {INPUT_SHAPE[1]} {INPUT_SHAPE[2]}"]) -> Float[Array, "2"]:
        for layer in self.layers:
            x = layer(x)
        return x

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
model = CNN(subkey)

import time
start_time = time.time()
test = jax.random.normal(key, (3, 28, 28))
print(test)
out = model(test)
print(out)
print("Execution time: %s seconds" % (time.time() - start_time))

start_time = time.time()
test = jax.random.normal(key, (3, 28, 28))
print(test)
out = model(test)
print(out)
print("Execution time: %s seconds" % (time.time() - start_time))
