import jax
import jax.numpy as jnp
import numpy as np
from timeit import timeit

def f(x):
  return -4*x*x*x + 9*x*x + 6*x - 3

x = np.random.randn(10000, 10000)

# Measure execution time using NumPy
numpy_time = timeit(lambda: f(x), number=10)
print("NumPy:", numpy_time)

# Measure execution time using JAX with JIT
jax_func = jax.jit(f)
jax_time = timeit(lambda: jax_func(jnp.array(x)), number=10)
print("JAX + JIT:", jax_time)