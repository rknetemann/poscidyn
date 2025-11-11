from abc import ABC, abstractmethod

from .. import constants as const

class AbstractExcitation(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def to_dtype(self, dtype):
        pass
    