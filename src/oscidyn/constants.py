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

MAXIMUM_ORDER_SUBHARMONICS = 10
MAXIMUM_ORDER_SUPERHARMONICS = 10

DT_MIN_FACTOR = 2000
DT_MAX_FACTOR = 200

MAX_SHOOTING_ITERATIONS = 20

PLOT_GRID = True

N_PERIODS_TO_RETAIN = 10
SAFETY_FACTOR_T_STEADY_STATE = 1.1
