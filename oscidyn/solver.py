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
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, max_periods: int = 512, rtol: float = 1e-6, atol: float = 1e-6):
        super().__init__(n_time_steps)
        self.max_steps = max_steps
        self.max_periods = max_periods
        self.rtol = rtol
        self.atol = atol
    
    def solve(self,
              model: AbstractModel, 
              driving_frequency: jax.Array, # Shape: (1,)
              driving_amplitude:  jax.Array, # Shape: (n_modes,)
              initial_condition:  jax.Array, # Shape: (2 * n_modes,)
              ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Integrate one drive window at a time until the RMS displacement
        between successive periods changes by less than `self.rtol` (relative)
        and `self.atol` (absolute).  Works with `vmap`/`jit`.
        """
        
        # ensure driving frequency is non-zero
        if jnp.any(driving_frequency == 0):
            raise ValueError("SteadyStateSolver is not compatible with zero driving frequency. Use StandardSolver for zero frequency cases (free vibration).")

        drive_period = 2.0 * jnp.pi / driving_frequency
        solve_window =  drive_period * const.MAXIMUM_ORDER_SUBHARMONICS # ASSUMPTION: MAXIMUM_ORDER_SUBHARMONICS means that we can check for subharmonics of order MAXIMUM_ORDER_SUBHARMONICS maximum
        n_steps_period = self.n_time_steps
        n_modes = model.n_modes
        state_dim = 2 * n_modes # (displacement + velocity) for each mode

        delta_ts = jnp.linspace(0.0, solve_window, n_steps_period) # Shape: (n_steps_period,)

        # Create arrays to store all periods (up to max_periods)
        all_t = jnp.zeros((self.max_periods, n_steps_period))
        all_y = jnp.zeros((self.max_periods, n_steps_period, state_dim))

        # carry = (t0, y0, prev_rms, delta_rms, window, all_t, all_y)
        carry0 = (
            jnp.array(0.0, dtype=initial_condition.dtype),  # t0
            initial_condition,                              # y0
            jnp.ones(n_modes, dtype=initial_condition.dtype) * jnp.inf,  # prev_rms
            jnp.ones(n_modes, dtype=initial_condition.dtype) * jnp.inf,  # delta_rms
            jnp.ones(n_modes, dtype=initial_condition.dtype) * jnp.inf,  # prev_amp
            jnp.ones(n_modes, dtype=initial_condition.dtype) * jnp.inf,  # delta_amp
            jnp.array(0, dtype=jnp.int32),                     # window
            all_t,
            all_y,
        )


        def solve_periods(carry):
            t0, y0, prev_rms, _delta_rms, prev_amp, _delta_amp, window, all_t, all_y = carry
            ts = t0 + delta_ts

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(model.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=t0,
                t1=t0 + solve_window,
                dt0=None,
                y0=y0,
                max_steps=self.max_steps,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
                args=(driving_frequency, driving_amplitude),
                throw=False,                      
            )

            time = sol.ts # Shape: (n_steps_period,)
            displacement = sol.ys[:, :n_modes] # Shape: (n_steps_period, n_modes)
            velocity = sol.ys[:, n_modes:]  # Shape: (n_steps_period, n_modes)

            # compute RMS displacement and maximum amplitude over this window
            rms = jnp.sqrt(jnp.mean(displacement**2, axis=0))  
            amp = jnp.max(jnp.abs(displacement), axis=0)

            delta_rms = jnp.abs(rms - prev_rms)
            delta_amp = jnp.abs(amp - prev_amp)

            # store this window's results
            all_t = all_t.at[window].set(sol.ts)
            all_y = all_y.at[window].set(sol.ys)

            # jax.debug.print("SteadyStateSolver: window {window} done, "
            #                 "rms={rms:.3e}, amp={amp:.3e}, delta_rms={delta_rms:.3e}, delta_amp={delta_amp:.3e}",
            #                 window=window, rms=rms, amp=amp, delta_rms=delta_rms, delta_amp=delta_amp)

            # update carry for next window
            return (t0 + solve_window,
                sol.ys[-1],
                rms,  delta_rms,
                amp,  delta_amp,
                window + 1,
                all_t, all_y)

        def steady_state_cond(carry):
            (_t0, _y0,
            prev_rms, delta_rms,
            prev_amp, delta_amp,
            window, *_ ) = carry

            eps = 1e-12
            rel_rms = delta_rms / jnp.maximum(prev_rms, eps)
            rel_amp = delta_amp / jnp.maximum(prev_amp, eps)

            rms_ok = (rel_rms < self.rtol) | (delta_rms < self.atol)
            amp_ok = (rel_amp < self.rtol) | (delta_amp < self.atol)

            all_modes_converged = jnp.all(rms_ok & amp_ok)

            MIN_PERIODS = 3
            converged = all_modes_converged & (window >= MIN_PERIODS) & (window > 0)

            keep_going = jnp.logical_and(~converged, window < self.max_periods)
            return keep_going
        
        (t0_f, y0_f,
        rms_f,  delta_rms_f,
        amp_f,  delta_amp_f,
        periods_f,
        all_t_f, all_y_f) = jax.lax.while_loop(
            steady_state_cond, solve_periods, carry0
        )

        # sanity: raise if we never converged
        if self.max_periods and isinstance(periods_f, jax.Array):
            # (inside JIT this becomes a host callback)
            def _host_assert(periods_val):
                if periods_val == self.max_periods:
                    raise RuntimeError(
                        f"Steady‑state not reached after {self.max_periods} periods "
                        f"(rtol={self.rtol}, atol={self.atol}, "
                        f"last Δrms={delta_rms_f}, Δamp={delta_amp_f})."
                    )
                return ()
            jax.debug.callback(_host_assert, periods_f, ordered=True)

        # Get only the valid periods (the ones we actually computed)
        valid_t = all_t_f[:periods_f]
        valid_y = all_y_f[:periods_f]
        
        # Reshape to flatten all periods into a single timeline
        time = valid_t.reshape(-1)
        y_all = valid_y.reshape(-1, state_dim)

        displacements = y_all[:, :n_modes]
        velocities = y_all[:, n_modes:]

        return time, displacements, velocities