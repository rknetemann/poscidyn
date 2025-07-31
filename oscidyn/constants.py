# constants.py
from enum import Enum, auto

class SweepDirection(Enum):
    FORWARD  = auto()
    BACKWARD = auto()

class ResponseType(Enum):
    FrequencyResponse = auto()
    TimeResponse      = auto()
    
N_COARSE_DRIVING_FREQUENCIES = 10
N_COARSE_DRIVING_AMPLITUDES = 10
N_COARSE_INITIAL_DISPLACEMENTS= 10
N_COARSE_INITIAL_VELOCITIES = 10

MAXIMUM_ORDER_SUBHARMONICS = 10
MAXIMUM_ORDER_SUPERHARMONICS = 10

N_PERIODS_TO_RETAIN = 5
MIN_WINDOWS = 3  # Minimum number of windows to consider for convergence

