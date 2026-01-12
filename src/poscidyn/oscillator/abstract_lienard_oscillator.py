import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
from typing import Any, Callable
from abc import abstractmethod
from equinox import filter_jit
import jax

from .abstract_oscillator import AbstractOscillator

class AbstractLienardOscillator (AbstractOscillator):
    def __init__(self):
        pass

    def __init_subclass__(cls):
        super().__init_subclass__()

    @filter_jit
    def rhs(self, t: Float, state: Array,
            *
            args: PyTree):
        """Right-hand side of the Lienard oscillator equations

        Wikipedia: [https://en.wikipedia.org/wiki/Li%C3%A9nard_equation](https://en.wikipedia.org/wiki/Li%C3%A9nard_equation)

        Args:
            t (float): Time
            state (Array): State vector
            args (PyTree): Additional arguments
        """
        x, dx_dt = jnp.split(state, 2)
        d2x_dt2 = - self.f(t, x, *args) * dx_dt - self.g(t, x, *args)

        return jnp.concatenate([dx_dt, d2x_dt2])
    
    @abstractmethod
    def f(self, t, state, args):
        """ Damping term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)
        """
        pass

    @abstractmethod
    def g(self, t, state, args):
        """ Stiffness term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)
        """
        pass