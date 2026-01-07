from __future__ import annotations
from dataclasses import field, dataclass
from abc import ABC, abstractmethod

oscillator = lambda cls: dataclass(eq=False, kw_only=True)(cls)

@oscillator
class AbstractOscillator(ABC):
    def __post_init__(self):
        pass
    
    @abstractmethod
    def to_dtype(self, dtype):
        pass

    @abstractmethod
    def f(self, t, state, args):
        pass

    @abstractmethod
    def f_y(self, t, state, args):
        pass
    