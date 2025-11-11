from .abstract_sweep import AbstractSweep
from .sweep_directions import AbstractSweepDirection

from .sweep_directions import Forward, Backward

class NearestNeighbourSweep(AbstractSweep):
    def __init__(self, sweep_direction: AbstractSweepDirection=Forward()):
        super().__init__()
        self.sweep_direction = sweep_direction
        
    def to_dtype(self, dtype):
        return NearestNeighbourSweep(
            sweep_direction=self.sweep_direction
        )
        
    def sweep(self,):
        pass