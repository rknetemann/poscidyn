from jax import random
import jax.numpy as jnp
import timeit

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

key = random.key(1701)
x = random.normal(key, (1_000_000,))
print(timeit.timeit(lambda: selu(x).block_until_ready(), number=100))