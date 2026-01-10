## Ease of Use  
Python is well known for its accessibility and readability, and this pythonic philosophy is consistently reflected throughout the package. Poscidyn is fully compatible with NumPy arrays, allowing practitioners to immediately leverage their existing knowledge and workflows without additional overhead.

## Performance  
Poscidyn is implemented entirely using JAX, which enables all core functions to be seamlessly `jitted` and `vmapped`. While these features are not required for basic usage, users are strongly encouraged to consult the [JAX documentation](https://docs.jax.dev/en/latest/) to fully unlock Poscidyn’s performance potential.

In practical terms, this design allows time-domain simulations and frequency sweeps to be batched and executed at scale very easily. As a result, Poscidyn is particularly well suited for applications where computational speed is essential.

## Connection to machine learning
Because Poscidyn is highly optimized for performance, it enables the efficient generation of large-scale dynamical datasets. Opening up a whole new world of data-driven applications such as training supervised learning models.

