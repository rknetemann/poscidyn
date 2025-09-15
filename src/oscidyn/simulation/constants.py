# constants.py
from enum import Enum, auto
import jax

class SweepDirection(Enum):
    FORWARD  = auto()
    BACKWARD = auto()

class Precision(Enum):
    SINGLE = auto()
    DOUBLE = auto()

class ResponseType(Enum):
    FrequencyResponse = auto()
    TimeResponse      = auto()

N_COARSE_DRIVING_FREQUENCIES = 50
N_COARSE_DRIVING_AMPLITUDES = 20
N_COARSE_INITIAL_DISPLACEMENTS= 5
N_COARSE_INITIAL_VELOCITIES = 5

MAXIMUM_ORDER_SUBHARMONICS = 10
MAXIMUM_ORDER_SUPERHARMONICS = 10

N_PERIODS_TO_RETAIN = 5
MIN_WINDOWS = 3  # Minimum number of windows to consider for convergence

PLOT_GRID = True

XLA_PYTHON_CLIENT_MEM_FRACTION = 0.85
