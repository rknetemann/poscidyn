# solver.py
import jax
import jax.numpy as jnp
#jax.config.update("jax_enable_x64", True)
import diffrax
from functools import partial
import numpy as np
import os

from .models import AbstractModel
from . import constants as const 

class AbstractSolver:
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-6, max_steps: int = 4096, progress_bar: bool = True):

        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.progress_bar = progress_bar
    
    def solve(self, 
              model: AbstractModel,
              t0: float, 
              t1: float, 
              ts: jax.Array,
              y0: jax.Array,
              driving_frequency: float, 
              driving_amplitude: float, 
              ) -> diffrax.Solution:
                
        progress_meter = diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter()
                
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            throw=True,
            progress_meter=progress_meter,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0),
            args=(driving_amplitude, driving_frequency),
        )
        return sol

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
    
class ShootingSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096,
                 rtol: float = 1e-6, atol: float = 1e-6, progress_bar: bool = False):
        super().__init__(rtol=rtol, atol=atol, max_steps=max_steps)
        self.n_time_steps = n_time_steps
        self.progress_bar = progress_bar

    def __call__(self,
                 model: AbstractModel,
                 driving_frequency: jax.Array,   # shape (1,) or scalar
                 driving_amplitude: jax.Array,   # shape (n_modes,) — here we assume 1 mode
                 initial_condition: jax.Array,   # shape (2,) for single-mode Duffing
                 response: const.ResponseType,
                 time_shift: float = 0.0):
        
        T = const.MAXIMUM_ORDER_SUBHARMONICS * 2.0 * jnp.pi / driving_frequency  # Period of oscillation including subharmonics
        t0 = 0.0 + time_shift
        t1 = T + time_shift
        ts = jnp.linspace(0.0, T, self.n_time_steps)
                
        TOL = 1e-4

        def _converged_cond(carry):
            y0, yT, ys, multipliers = carry
            shooting_residual = jnp.linalg.norm(F, ord=jnp.inf)
            jax.debug.print("Shooting residual: {}", shooting_residual)
            return shooting_residual > TOL
        
        def _newton_shooting(y0):
            X0 = jnp.eye(2).reshape(-1)  # Initial condition for the state transition matrix (flattened)
            y0_aug = jnp.hstack([y0, X0])

            ys_aug = self._solve(model, t0, t1, ts, y0_aug, driving_frequency, driving_amplitude).ys
            ys = ys_aug[:, :2]
            yT = ys_aug[-1, :2]
            XT = ys_aug[-1, 2:].reshape(2, 2)
            
            F = yT - y0
            J = XT - jnp.eye(2)
            
            r = jnp.linalg.norm(F, ord=jnp.inf)
            
            mu = jnp.linalg.eigvals(XT)
            
            return y0, yT, ys, F, J, r, mu

        def _shooting_iteration(carry):
            y0, yT, ys, multipliers = carry

            y0, yT, ys, F, J, r, mu = _newton_shooting(y0)
                      
            dy0 = jnp.linalg.solve(J, -F)
            
            def _line_search_cond(carry):
                i, lam, y0, dy0, r, done = carry
                return jnp.logical_and(~done, i < 8)

            def _line_search_iteration(carry):
                i, lam, y0, dy0, r, done = carry

                y0_try = y0 + lam * dy0
                y0_try, yT, ys, F, J, r_try, mu = _newton_shooting(y0_try)

                success = r_try < 0.7 * r

                # Accept step if successful; otherwise halve step size and continue
                y0 = jnp.where(success, y0_try, y0)
                done = jnp.logical_or(done, success)
                lam = jnp.where(success, lam, lam * 0.5)

                return (i + 1, lam, y0, dy0, r, done)          
            
            init_carry = (
                jnp.array(0, dtype=jnp.int32),           # i
                jnp.array(1.0, dtype=y0.dtype),          # lam
                y0,                                      # y0
                dy0,                                     # dy0
                r,                                       # r
                jnp.array(False),                        # done
            )

            i, lam, y0_after_ls, dy0, r, done = jax.lax.while_loop(
                _line_search_cond, _line_search_iteration, init_carry
            )

            # for-else fallback: if never succeeded, use last resort y0 + dy0
            y0 = jnp.where(done, y0_after_ls, y0 + dy0)  
                        
            return y0, yT, ys, mu
        
        init_carry = (
            initial_condition,
            jnp.zeros((self.n_time_steps, 2 + 4)),
            jnp.full((2,), jnp.inf),
            jnp.zeros((2,), dtype=jnp.complex64),
        )
        
        y0, ys, F, multipliers = jax.lax.while_loop(_converged_cond, _shooting_iteration, init_carry)
        
        return ts, ys[:, :2]  # Return time and state (displacement and velocity)

    def _solve(self,
               model: AbstractModel,
               t0: float,
               t1: float,
               ts: jax.Array,
               y0: jax.Array,
               driving_frequency: float,
               driving_amplitude: jax.Array) -> diffrax.Solution:

        @jax.jit
        def aug_rhs(t, y_aug, args):
            y  = y_aug[:2]
            X  = y_aug[2:].reshape(2, 2)

            f = model.f(t, y, args)      
            f_y  = model.f_y(t, y, args)

            dydt = f
            dXdt = f_y @ X
            return jnp.hstack([dydt, dXdt.reshape(-1)])

        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(aug_rhs),
            solver=diffrax.Tsit5(),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            throw=True,
            progress_meter=diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0
            ),
            args=(driving_amplitude, driving_frequency),
        )
        return sol

class SteadyStateSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, rtol: float = 1e-6, atol: float = 1e-6, 
                 ss_rtol: float = 1e-3, ss_atol: float = 1e-6, max_windows: int = 512):

        super().__init__(rtol=rtol, atol=atol, max_steps=max_steps)
        self.n_time_steps = n_time_steps 
        self.max_windows = max_windows
        self.ss_rtol = ss_rtol
        self.ss_atol = ss_atol

    def __call__(self,
              model: AbstractModel, 
              driving_frequency: jax.Array, # Shape: (1,)
              driving_amplitude:  jax.Array, # Shape: (n_modes,)
              initial_condition:  jax.Array, # Shape: (2 * n_modes,) # TO DO: Initial conditions not yet used
              response: const.ResponseType,
              time_shift: float = 0.0, # TO DO: Not yet used
              ):
        
        if response == const.ResponseType.FrequencyResponse:
            windows_to_save = 1
        elif response == const.ResponseType.TimeResponse:
            windows_to_save = self.max_windows

        drive_period = 2.0 * jnp.pi / driving_frequency
        solve_window =  drive_period * const.MAXIMUM_ORDER_SUBHARMONICS # ASSUMPTION: MAXIMUM_ORDER_SUBHARMONICS means that we can check for subharmonics of order MAXIMUM_ORDER_SUBHARMONICS maximum
        n_steps_window = self.n_time_steps
        n_modes = model.n_modes
        state_dim = 2 * n_modes

        # We only save one window 
        ts = jnp.zeros((windows_to_save, n_steps_window))
        ys = jnp.zeros((windows_to_save, n_steps_window, state_dim))
        rms = jnp.full((n_modes,), jnp.inf)
        max = jnp.full((n_modes,), jnp.inf)
        prev_rms = jnp.full((n_modes,), jnp.inf)
        prev_max = jnp.full((n_modes,), jnp.inf)
        win_idx = 0

        init_window = (ts, ys, rms, max, prev_rms, prev_max, win_idx)

        # vmap-compatible
        def solve_windows(window):
            ts, ys, rms, max, _prev_rms, _prev_max, win_idx = window

            previous_window_idx = jax.lax.cond(windows_to_save == 1, lambda: 0, lambda:  win_idx)
            t0 = ts[previous_window_idx, -1]
            t1 = t0 + solve_window
            y0 = ys[previous_window_idx, -1]

            # Generate a 1D array of time points for this window
            ts_window = jnp.linspace(t0, t1, n_steps_window)

            sol = self.solve(model=model, t0=t0, t1=t1, ts=ts_window, y0=y0, driving_frequency=driving_frequency, driving_amplitude=driving_amplitude)

            ts_window = sol.ts # Shape: (n_steps_window,)
            ys_window = sol.ys # Shape: (n_steps_window, state_dim)

            prev_rms = rms
            prev_max = max
            rms = jnp.sqrt(jnp.mean(ys_window[:, :n_modes]**2, axis=0))
            max = jnp.max(jnp.abs(ys_window[:, :n_modes]), axis=0)

            win_idx += 1 
            current_window_idx = jax.lax.cond(windows_to_save == 1, lambda: 0, lambda: win_idx)
            ts = ts.at[current_window_idx].set(ts_window)
            ys = ys.at[current_window_idx].set(ys_window)           

            return ts, ys, rms, max, prev_rms, prev_max, win_idx
        
        # Keep solving solve_windows() until we reach steady state or hit max_windows
        (ts, ys, _rms, max, _prev_rms, _prev_max, _n_windows) = jax.lax.while_loop(self._steady_state_cond, solve_windows, init_window)       
        return ts, ys
        
        
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