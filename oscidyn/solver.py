# solver.py
import jax
import jax.numpy as jnp
import diffrax
from functools import partial

from .models import AbstractModel
from . import constants as const 

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')

class AbstractSolver:
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, 
                 rtol: float = 1e-4, atol: float = 1e-6):
        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.rtol = rtol
        self.atol = atol
    
    def solve(self, 
              model: AbstractModel,
              t0: float, 
              t1: float, 
              y0: jax.Array,
              driving_frequency: float, 
              driving_amplitude: float, 
              ) -> diffrax.Solution:
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            throw=False,
            #progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, self.n_time_steps)),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_frequency, driving_amplitude),
        )
        return sol

class FixedTimeSolver(AbstractSolver):
    def __init__(self, t1: float, t0: float = 0, n_time_steps: int = 2000, max_steps: int = 4096, 
                 rtol: float = 1e-4, atol: float = 1e-6):
        super().__init__(n_time_steps, max_steps, rtol, atol)
        self.t0 = t0
        self.t1 = t1

    def __call__(self, model: AbstractModel, 
              driving_frequency: jax.Array, 
              driving_amplitude: jax.Array, 
              initial_condition: jax.Array,
              response: const.ResponseType
              ) -> tuple[jax.Array, jax.Array, jax.Array]:

        sol = self.solve(model=model, t0=self.t0, t1=self.t1, y0=initial_condition, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

        time = sol.ts # Shape: (n_steps,)
        displacements = sol.ys[:, : model.n_modes] # Shape: (n_steps, n_modes)
        velocities = sol.ys[:, model.n_modes :] # Shape: (n_steps, n_modes)
        
        return time, displacements, velocities

class SteadyStateSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, rtol: float = 1e-6, atol: float = 1e-6, 
                 ss_rtol: float = 1e-3, ss_atol: float = 1e-6, max_windows: int = 512):
        super().__init__(n_time_steps, max_steps, rtol, atol)
        self.max_windows = max_windows
        self.ss_rtol = ss_rtol
        self.ss_atol = ss_atol

    def __call__(self,
              model: AbstractModel, 
              driving_frequency: jax.Array, # Shape: (1,)
              driving_amplitude:  jax.Array, # Shape: (n_modes,)
              initial_condition:  jax.Array, # Shape: (2 * n_modes,) # TO DO: Initial conditions not yet used
              response: const.ResponseType
              ):

        if response == const.ResponseType.FrequencyResponse:
            return self.solve_frequency_response(model, driving_frequency, driving_amplitude)
        elif response == const.ResponseType.TimeResponse:
            return self.solve_time_response(model, driving_frequency, driving_amplitude)
        
    @partial(jax.jit, static_argnames=['self'])
    def _steady_state_cond(self, window):
        (_ts, _ys, rms, max, prev_rms, prev_max, win_idx) = window          

        delta_rms = jnp.abs(prev_rms - rms)
        delta_max = jnp.abs(prev_max - max)

        eps = 1e-12
        rel_rms = delta_rms / jnp.maximum(prev_rms, eps)
        rel_max = delta_max / jnp.maximum(prev_max, eps)

        rms_ok = (rel_rms < self.ss_rtol) | (delta_rms < self.ss_atol)
        max_ok = (rel_max < self.ss_rtol) | (delta_max < self.ss_atol)

        all_modes_converged = jnp.all(rms_ok & max_ok)

        converged = all_modes_converged & (win_idx >= const.MIN_WINDOWS) & (win_idx > 0)

        keep_going = jnp.logical_and(~converged, win_idx < self.max_windows)
        return keep_going

    def solve_time_response(self, model: AbstractModel, driving_frequency: float, driving_amplitude: float):
        drive_period = 2.0 * jnp.pi / driving_frequency
        solve_window =  drive_period * const.MAXIMUM_ORDER_SUBHARMONICS # ASSUMPTION: MAXIMUM_ORDER_SUBHARMONICS means that we can check for subharmonics of order MAXIMUM_ORDER_SUBHARMONICS maximum
        n_steps_window = self.n_time_steps
        n_modes = model.n_modes
        state_dim = 2 * n_modes

        # Create arrays to store all periods (up to max_windows)
        ts = jnp.zeros((self.max_windows, n_steps_window))
        ys = jnp.zeros((self.max_windows, n_steps_window, state_dim))
        rms = jnp.full((n_modes,), jnp.inf)  # RMS of displacements for each mode
        max = jnp.full((n_modes,), jnp.inf)  # Max of displacements for each mode
        prev_rms = jnp.full((n_modes,), jnp.inf)  # Previous RMS of displacements for each mode
        prev_max = jnp.full((n_modes,), jnp.inf)  # Previous Max of displacements for each mode
        win_idx = 0
        
        init_window = (ts, ys, rms, max, prev_rms, prev_max, win_idx)

        # vmap-compatible
        def solve_windows(window):
            ts, ys, rms, max, _prev_rms, _prev_max, win_idx = window

            t0 = ts[win_idx - 1, -1]
            t1 = t0 + solve_window
            y0 = ys[win_idx - 1, -1]

            sol = self.solve(model=model, t0=t0, t1=t1, y0=y0, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

            ts_window = sol.ts # Shape: (n_steps_window,)
            ys_window = sol.ys # Shape: (n_steps_window, state_dim)

            prev_rms = rms
            prev_max = max
            rms = jnp.sqrt(jnp.mean(ys_window[:, :n_modes]**2, axis=0))
            max = jnp.max(jnp.abs(ys_window[:, :n_modes]), axis=0)

            ts = ts.at[win_idx].set(ts_window)
            ys = ys.at[win_idx].set(ys_window)

            win_idx += 1 

            return ts, ys, rms, max, prev_rms, prev_max, win_idx

        # Keep solving solve_windows() until we reach steady state or hit max_windows
        (ts, ys, rms, max, prev_rms, prev_max, n_windows) = jax.lax.while_loop(self._steady_state_cond, solve_windows, init_window)       
        
        time = ts.flatten()
        # Reshape displacements and velocities to match flattened time
        displacements = ys[:, :, :n_modes].reshape(-1, n_modes)  # Shape: (max_windows * n_steps_window, n_modes)
        velocities = ys[:, :, n_modes:].reshape(-1, n_modes)  # Shape: (max_windows * n_steps_window, n_modes)

        return time, displacements, velocities

    def solve_frequency_response(self, model: AbstractModel, driving_frequency: float, driving_amplitude: float):
        drive_period = 2.0 * jnp.pi / driving_frequency
        solve_window =  drive_period * const.MAXIMUM_ORDER_SUBHARMONICS # ASSUMPTION: MAXIMUM_ORDER_SUBHARMONICS means that we can check for subharmonics of order MAXIMUM_ORDER_SUBHARMONICS maximum
        n_steps_window = self.n_time_steps
        n_modes = model.n_modes
        state_dim = 2 * n_modes

        # We only save one window 
        ts = jnp.zeros((1, n_steps_window))
        ys = jnp.zeros((1, n_steps_window, state_dim))
        rms = jnp.full((n_modes,), jnp.inf)
        max = jnp.full((n_modes,), jnp.inf)
        prev_rms = jnp.full((n_modes,), jnp.inf)
        prev_max = jnp.full((n_modes,), jnp.inf)
        win_idx = 0

        init_window = (ts, ys, rms, max, prev_rms, prev_max, win_idx)

        # vmap-compatible
        def solve_windows(window):
            ts, ys, rms, max, _prev_rms, _prev_max, win_idx = window

            # TO DO: We can probably optimize ts for frequency solves, because we only need the last value of the window
            t0 = ts[0,-1]
            t1 = t0 + solve_window
            y0 = ys[0,-1]

            sol = self.solve(model=model, t0=t0, t1=t1, y0=y0, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

            ts_window = sol.ts # Shape: (n_steps_window,)
            ys_window = sol.ys # Shape: (n_steps_window, state_dim)

            prev_rms = rms
            prev_max = max
            rms = jnp.sqrt(jnp.mean(ys_window[:, :n_modes]**2, axis=0))
            max = jnp.max(jnp.abs(ys_window[:, :n_modes]), axis=0)

            ts = ts.at[win_idx].set(ts_window)
            ys = ys.at[win_idx].set(ys_window)

            win_idx += 1 

            return ts, ys, rms, max, prev_rms, prev_max, win_idx
        
        # Keep solving solve_windows() until we reach steady state or hit max_windows
        (ts, ys, _rms, max, _prev_rms, _prev_max, _n_windows) = jax.lax.while_loop(self._steady_state_cond, solve_windows, init_window)       
        
        amplitude = max # Shape: (n_modes,)
        phase = max # Shape: (n_modes,)

        return amplitude, phase