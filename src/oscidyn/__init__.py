from .constants import *
from .solver.abstract_solver import *
from .solver.single_shooting_solver import *
from .solver.multiple_shooting_solver import *
from .solver.collocation_solver import *
from .solver.fixed_time_solver import *
from .solver.multistart.linear_response_multistart import *

from .model.abstract_model import *
from .model.base_duffing_oscillator_model import *

from .frequency_sweep import frequency_sweep
from .time_response import time_response

from .utils.plotting import *

from .solver.utils.coarse_grid import gen_coarse_grid_1, gen_grid_2

"""
A Python toolkit for simulating and visualizing nonlinear oscillators using experimentally realistic setups, supporting both time- and frequency-domain analyses.
"""