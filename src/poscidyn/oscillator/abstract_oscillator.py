from __future__ import annotations
from jaxtyping import Float, Array, PyTree
from abc import abstractmethod, ABC

class AbstractOscillator (ABC):
    def __init__(self):
        pass

    def f_i(self, t: Float, y: Array, args: PyTree):
        """Internal forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): State vector
            args (PyTree): Additional arguments
        """
        pass