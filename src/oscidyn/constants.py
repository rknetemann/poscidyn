# constants.py
from enum import Enum, auto

class SweepDirection(Enum):
    FORWARD  = auto()
    BACKWARD = auto()
    
N_COARSE_DRIVING_FREQUENCIES = 100   
N_COARSE_DRIVING_AMPLITUDES = 5
N_COARSE_INITIAL_DISPLACEMENTS= 10
N_COARSE_INITIAL_VELOCITIES = 10

