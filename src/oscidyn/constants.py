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

N_COARSE_DRIVING_FREQUENCIES = 51 # Odd number to include middle frequency
N_COARSE_DRIVING_AMPLITUDES = 5
N_COARSE_INITIAL_DISPLACEMENTS= 21
N_COARSE_INITIAL_VELOCITIES = 21
N_COARSE_INITIAL_CONDITIONS_OFFSET_FACTOR = 2 

MAXIMUM_ORDER_SUBHARMONICS = 10
MAXIMUM_ORDER_SUPERHARMONICS = 10

DT_MIN_FACTOR = 2000
DT_MAX_FACTOR = 200

MAX_SHOOTING_ITERATIONS = 20

N_PERIODS_TO_RETAIN = 5
MIN_WINDOWS = 3  # Minimum number of windows to consider for convergence

PLOT_GRID = True

SAFETY_FACTOR_T_STEADY_STATE = 1.5
SAFETY_FACTOR_T_WINDOW = 2.0

STEADY_STATE_TOLERANCE = 1e-3

XLA_PYTHON_CLIENT_MEM_FRACTION = 0.95
