import jax
import jax.numpy as jnp
import diffrax

from .abstract_solver import AbstractSolver
from ..models.abstract_model import AbstractModel
from .. import constants as const 

class FixedTimeSolver(AbstractSolver):
    def __init__(self, duration: float, n_time_steps: int = None,
                 rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 4096):

        super().__init__(rtol, atol, max_steps)
        self.duration = duration
        self.n_time_steps = n_time_steps

    def __call__(self, model: AbstractModel, 
              driving_frequency: jax.Array, 
              driving_amplitude: jax.Array, 
              initial_condition: jax.Array,
              response: const.ResponseType,
              time_shift: float = 0.0,
              ):
        
        t0 = 0.0 + time_shift
        t1 = t0 + self.duration

        ts = jnp.linspace(t0, t1, self.n_time_steps)

        sol = self.solve(model=model, t0=t0, t1=t1, ts=ts, y0=initial_condition, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

        ts = sol.ts  # Shape: (n_steps,)
        ys = sol.ys  # Shape: (n_steps, state_dim)

        return ts, ys