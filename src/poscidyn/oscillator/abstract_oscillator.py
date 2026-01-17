from __future__ import annotations
from dataclasses import field, dataclass
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
from typing import Any, Callable
from abc import abstractmethod, ABC
from equinox import filter_jit

from ..excitation.abstract_excitation import AbstractExcitation

oscillator = lambda cls: dataclass(eq=False, kw_only=True)(cls)

@oscillator
class AbstractOscillator (ABC):
    excitation: AbstractExcitation = None

    def __init__(self):
        pass

    def __init_subclass__(cls):
        super().__init_subclass__()

    #@filter_jit
    def rhs(self, t: Float, state: Array, args: PyTree):
        """Right-hand side of the abstract oscillator equations
        Args:
            t (float): Time
            state (Array): State vector
            args (PyTree): Additional arguments
        """

        if self.excitation == None:
            raise ValueError("Excitation function is not set in the oscillator.")

        x, dx_dt = jnp.split(state, 2)
        damping_term = self.damping_term(t, state, args)
        stiffness_term = self.stiffness_term(t, state, args)
        d2x_dt2 = -  damping_term - stiffness_term + self.excitation.direct_drive(t, x, args)
        return jnp.concatenate([dx_dt, d2x_dt2])
    
    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Number of degrees of freedom"""
        pass
    
    @abstractmethod
    def damping_term(self, t, state, args):
        """ Damping term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)
        """
        pass

    @abstractmethod
    def stiffness_term(self, t, state, args):
        """ Stiffness term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)
        """
        pass
