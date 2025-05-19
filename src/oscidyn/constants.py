from enum import Enum, auto

class Sweep(Enum):
    FORWARD  = auto()
    BACKWARD = auto()
    
class Damping(Enum):
    NONE = auto()
    LIGHTLY_DAMPED = auto()
    MODERATELY_DAMPED = auto()

F_OMEGA_HAT_COARSE_N = 50    
Y0_HAT_COARSE_N = 30
