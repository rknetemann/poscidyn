import jax

def random_uniform(key, shape, lo, hi):
    key, sub = jax.random.split(key)
    return jax.random.uniform(sub, shape, minval=lo, maxval=hi), key