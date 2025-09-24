import jax
import jax.numpy as jnp
import diffrax

from .abstract_solver import AbstractSolver
from ..models.abstract_model import AbstractModel
from .. import constants as const 

class FixedTimeSteadyStateSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = None, ss_tol:float = 1e-3,
                 rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 4096, progress_bar: bool = True):
        
        super().__init__(rtol=rtol, atol=atol, max_steps=max_steps, progress_bar=progress_bar)
        self.n_time_steps = n_time_steps # Can be None, in which case it will be calculated based on the driving frequency 
        
        self.ss_tol = ss_tol

    def _calculate_time_window(self, model: AbstractModel, driving_frequency) -> jax.Array:
        '''
        Calculate the minimum time window/ time period to capture all harmonics.
        '''
        time_window = const.MAXIMUM_ORDER_SUBHARMONICS * (2 * jnp.pi / jnp.max(driving_frequency)) * const.SAFETY_FACTOR_T_WINDOW 
         
        return time_window

    def __call__(self, model: AbstractModel, 
              driving_frequency: jax.Array, 
              driving_amplitude: jax.Array, 
              initial_condition: jax.Array,
              response: const.ResponseType,
              time_shift: float = 0.0,
              ):

        settling_time = model.t_steady_state(driving_frequency, self.ss_tol) # Shape: ()
        steady_state_window = self._calculate_time_window(model, driving_frequency) 
        t0 = 0.0 + time_shift # Time to start numerical integration
        t1 = settling_time + steady_state_window + time_shift # Time to end numerical integration

        ts = jnp.linspace(settling_time + time_shift, t1, self.n_time_steps) # Time steps to save

        sol = self.solve(model=model, t0=t0, t1=t1, ts=ts, y0=initial_condition, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

        ts = sol.ts  # Shape: (n_steps,)
        ys = sol.ys  # Shape: (n_steps, state_dim)

        return ts, ys