from __future__ import annotations
from dataclasses import field, dataclass
import jax.numpy as jnp
from jaxtyping import Float, Array, PyTree
from typing import Any, Callable
from abc import abstractmethod, ABC
from equinox import filter_jit

oscillator = lambda cls: dataclass(eq=False, kw_only=True)(cls)

@oscillator
class AbstractOscillator (ABC):
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
        damping_term = self.damping_term(t, x, *args) + self.parametric_damping_term(t, x, *args)
        stiffness_term = self.stiffness_term(t, x, *args) + self.parametric_stiffness_term(t, x, *args)
        d2x_dt2 = -  damping_term * dx_dt - stiffness_term + self.direct_drive_term(t, x, *args)
        return jnp.concatenate([dx_dt, d2x_dt2])
    
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

    def direct_drive_term(self, t, state, args):
        """ Direct drive function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)

        Defaults to zero if not overridden.

        Returns:
            float: Direct drive term
        """
        return jnp.zeros_like(state.size / 2)

    def parametric_damping_term(self, t, state, args):
        """ Parametric damping term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)

        Defaults to zero if not overridden.

        Returns:
            float: Parametric damping term
        """
        return jnp.zeros_like(state.size / 2)
    
    def parametric_stiffness_term(self, t, state, args):
        """ Parametric stiffness term function
        d2x/dt2 + f(x) dx/dt + g(x) = f(t)

        Defaults to zero if not overridden.

        Returns:
            float: Parametric stiffness term
        """
        return jnp.zeros_like(state.size / 2)
