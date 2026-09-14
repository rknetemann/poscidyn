from abc import ABC, abstractmethod

class AbstractSyntheticSweepDirection:
    def __init__(self):
        pass

class Forward(AbstractSyntheticSweepDirection):
    pass

class Backward(AbstractSyntheticSweepDirection):
    pass

class AbstractSyntheticSweep(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dtype(self, dtype):
        pass
    