from abc import ABC, abstractmethod

class AbstractSweep(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dtype(self, dtype):
        pass
    