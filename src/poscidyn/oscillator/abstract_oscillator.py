from __future__ import annotations
from dataclasses import field, dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
from typing import Any, Callable
from abc import abstractmethod, ABC
from equinox import filter_jit

from ..excitation.abstract_excitation import AbstractExcitation

class AbstractOscillator (ABC):
    
    def __init__(self):
        self.excitation: AbstractExcitation = None

    def __init_subclass__(cls):
        super().__init_subclass__()

    def f_i(self, t: Float, state: Array, args: PyTree):
        """Internal forces of the abstract oscillator equations of motion.

        Args:
            t (float): Time
            state (Array): State vector
            args (PyTree): Additional arguments
        """

        pass