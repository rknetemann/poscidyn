import jax
import jax.numpy as jnp

batch_of_params = jnp.array([0, 1, 2, 3, 4, 5])  # Example batch of parameters

def simulate(params):
    # Simulate some computation with the parameters
    some_result = params * 2  
    return some_result

# Parallelize across devices
parallel_simulate = jax.pmap(simulate, axis_name="devices")

# Split your batch for each device
n_devices = jax.device_count()
params_shards = jnp.array_split(batch_of_params, n_devices)

print("Parameters for each device:", params_shards)

# Put shards on GPUs and run
results = parallel_simulate(jnp.array(params_shards))

print("Results from each device:", results)
