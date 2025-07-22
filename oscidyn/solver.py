# solver.py
import jax
import jax.numpy as jnp
import diffrax

from .models import AbstractModel
from . import constants as const 

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')

class AbstractSolver:
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096):
        self.n_time_steps = n_time_steps
        self.max_steps = max_steps

class StandardSolver(AbstractSolver):
    def __init__(self, t_end: float, n_time_steps: int = 2000, max_steps: int = 4096):
        super().__init__(n_time_steps)
        self.t_end = t_end
        self.max_steps = max_steps

    def solve(self, model: AbstractModel, 
              driving_frequency: jax.Array, 
              driving_amplitude: jax.Array, 
              initial_condition: jax.Array
              ) -> tuple[jax.Array, jax.Array, jax.Array]:
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=self.t_end,
            dt0=None,
            max_steps=self.max_steps,
            y0=initial_condition,
            throw=True,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, self.t_end, self.n_time_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            args=(driving_frequency, driving_amplitude),
        )

        time = sol.ts # Shape: (n_steps,)
        displacements = sol.ys[:, : model.n_modes] # Shape: (n_steps, n_modes)
        velocities = sol.ys[:, model.n_modes :] # Shape: (n_steps, n_modes)
        
        return time, displacements, velocities

class SteadyStateSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, max_windows: int = 512, rtol: float = 1e-6, atol: float = 1e-6):
        super().__init__(n_time_steps)
        self.max_steps = max_steps
        self.max_windows = max_windows
        self.rtol = rtol
        self.atol = atol

    # jit-incompatible, vmap-compatible
    def solve(self,
              model: AbstractModel, 
              driving_frequency: jax.Array, # Shape: (1,)
              driving_amplitude:  jax.Array, # Shape: (n_modes,)
              initial_condition:  jax.Array, # Shape: (2 * n_modes,) # TO DO: Initial conditions not yet used
              only_amplitude: bool = False # TO DO: If True only save the amplitude of the steady state, not the full state
              ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Integrate one drive window at a time until the RMS displacement
        between successive periods changes by less than `self.rtol` (relative)
        and `self.atol` (absolute).  Works with `vmap`/`jit`.
        """
        
        drive_period = 2.0 * jnp.pi / driving_frequency
        solve_window =  drive_period * const.MAXIMUM_ORDER_SUBHARMONICS # ASSUMPTION: MAXIMUM_ORDER_SUBHARMONICS means that we can check for subharmonics of order MAXIMUM_ORDER_SUBHARMONICS maximum
        n_steps_window = self.n_time_steps
        n_modes = model.n_modes
        state_dim = 2 * n_modes # (displacement + velocity) for each mode

        # Create arrays to store all periods (up to max_windows)
        ts = jnp.zeros((self.max_windows, n_steps_window))
        ys = jnp.zeros((self.max_windows, n_steps_window, state_dim))
        win_idx = 0 

        init_window = (ts, ys, win_idx)

        # vmap-compatible
        def solve_windows(window):
            ts, ys, win_idx = window

            t0 = ts[win_idx - 1, -1]
            t1 = t0 + solve_window
            ts_window = jnp.linspace(t0, t1, n_steps_window) # Shape: (n_steps_window,), ts of current window
            y0 = ys[win_idx - 1, -1]  # Initial condition for the current window

            # jax.debug.print("Solving window {win_idx}: t0={t0}, t1={t1}, y0={y0}",
            #                 win_idx=win_idx, t0=t0, t1=t1, y0=y0)

            # vmap-compatible
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(model.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=t0,
                t1=t1,
                dt0=None,
                y0=y0,
                max_steps=self.max_steps,
                saveat=diffrax.SaveAt(ts=ts_window),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
                args=(driving_frequency, driving_amplitude),
                throw=False,                      
            )

            ts_window = sol.ts # Shape: (n_steps_window,)
            ys_window = sol.ys # Shape: (n_steps_window, state_dim)

            ts = ts.at[win_idx].set(ts_window)  # Store the time steps for this window
            ys = ys.at[win_idx].set(ys_window)  # Store the state for this window
            win_idx += 1  # Increment the window index

            return ts, ys, win_idx

        @jax.jit
        def steady_state_cond(window):
            (ts, ys, win_idx) = window

            # Initialize prev_rms and prev_max to +inf when win_idx == 0, else compute from the last window
            prev_rms, prev_max = jax.lax.cond(
                win_idx > 1,
                lambda idx: (
                    jnp.sqrt(jnp.mean(ys[idx - 2][:, :n_modes]**2, axis=0)),
                    jnp.max(jnp.abs(ys[idx - 2][:, :n_modes]), axis=0),
                ),
                lambda _: (
                    jnp.full((n_modes,), jnp.inf),
                    jnp.full((n_modes,), jnp.inf),
                ),
                operand=win_idx
            )

            rms = jnp.sqrt(jnp.mean(ys[win_idx - 1][:, :n_modes]**2, axis=0))
            max = jnp.max(jnp.abs(ys[win_idx - 1][:, :n_modes]), axis=0)

            delta_rms = jnp.abs(prev_rms - rms)
            delta_max = jnp.abs(prev_max - max)

            eps = 1e-12
            rel_rms = delta_rms / jnp.maximum(prev_rms, eps)
            rel_max = delta_max / jnp.maximum(prev_max, eps)

            # jax.debug.print(
            #     "win_idx: {win_idx}, "
            #     "prev_rms: {prev_rms}, rms: {rms}, rel_rms: {rel_rms}, "
            #     "prev_max: {prev_max}, max: {max}, rel_max: {rel_max}", 
            #     win_idx=win_idx,
            #     prev_rms=prev_rms, rms=rms, rel_rms=rel_rms,
            #     prev_max=prev_max, max=max, rel_max=rel_max
            # )

            rms_ok = (rel_rms < self.rtol) | (delta_rms < self.atol)
            max_ok = (rel_max < self.rtol) | (delta_max < self.atol)

            all_modes_converged = jnp.all(rms_ok & max_ok)

            MIN_PERIODS = 3
            converged = all_modes_converged & (win_idx >= MIN_PERIODS) & (win_idx > 0)

            keep_going = jnp.logical_and(~converged, win_idx < self.max_windows)
            return keep_going
        
        # Keep solving solve_windows() until we reach steady state or hit max_windows
        (ts, ys, n_windows) = jax.lax.while_loop(steady_state_cond, solve_windows, init_window)

        

        return time, displacements, velocities