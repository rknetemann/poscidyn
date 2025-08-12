import jax
import os
#jax.distributed.initialize()

print(os.environ.get("CUDA_VISIBLE_DEVICES"))

print("jax.device_count()", jax.device_count())
print("jax.local_device_count()", jax.local_device_count())
print("jax.devices()", jax.devices())