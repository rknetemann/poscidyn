from .constants import *
from .solver import *

from .examples.models import *

from .simulations.frequency_sweep import frequency_sweep, vmap_safe_frequency_sweep
from .simulations.time_response import time_response

from .utils.plotting import *

"""
A Python toolkit for simulating and visualizing nonlinear oscillators using experimentally realistic setups, supporting both time- and frequency-domain analyses.
"""

import jax

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')
jax.config.update('jax_compiler_enable_remat_pass', True)

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' 
