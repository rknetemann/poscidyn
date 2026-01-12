import math
import jax
import jax.numpy as jnp
from equinox import filter_jit
import diffrax
from jax import core as jax_core

from .abstract_solver import AbstractSolver
from ..oscillator.abstract_oscillator import AbstractOscillator
from ..multistart.abstract_multistart import AbstractMultistart
from ..excitation.abstract_excitation import AbstractExcitation
from ..sweep.abstract_sweep import AbstractSweep
from ..result.frequency_sweep_result import FrequencySweepResult

from .. import constants as const 

class TimeIntegrationSolver(AbstractSolver):
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, 
                 verbose: bool = False, throw: bool = False):
        
        self.max_steps = max_steps
        self.n_time_steps = n_time_steps
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.throw = throw

        self.model: AbstractOscillator = None
        self.multistarter: AbstractMultistart = None

    @staticmethod
    def _is_tracer(value) -> bool:
        """Check whether a value is being traced by JAX (e.g. inside vmap/jit)."""
        return isinstance(value, jax_core.Tracer)
    
    def _max_steps_budget(self, t_span: float, period: float, safety_factor: float = 2.0) -> int:
        """Infer a max_steps large enough for the requested horizon.

        We approximate the solver step size with the sampling interval
        (period / (n_time_steps * MAXIMUM_ORDER_SUPERHARMONICS)) and
        inflate by a safety factor so we don't trip the diffrax max_steps
        guard on long, low-frequency runs.
        """
        if self.n_time_steps is None:
            return self.max_steps

        # When traced (e.g. under vmap/jit), fall back to the configured cap
        # instead of trying to convert tracers to Python scalars.
        if self._is_tracer(t_span) or self._is_tracer(period):
            return self.max_steps

        steps_per_period = max(int(self.n_time_steps), 1) * const.MAXIMUM_ORDER_SUPERHARMONICS
        dt_est = float(period) / steps_per_period
        if not math.isfinite(dt_est) or dt_est <= 0.0:
            return self.max_steps

        est_steps = math.ceil((t_span / dt_est) * safety_factor)
        return max(self.max_steps, est_steps)

    def time_response(self,
                 f_omega: jax.Array,  
                 f_amp: jax.Array, 
                 x0: jax.Array,  
                 v0: jax.Array,
                **kwargs
                ):

        y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

        if self.n_time_steps is None and not self._is_tracer(f_omega):
            rtol = 0.01
            max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * jnp.max(f_omega)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component
            
            n_time_steps = int(math.ceil(float(one_period * sampling_frequency)))
            self.n_time_steps = n_time_steps
        elif self.n_time_steps is None:
            raise ValueError("n_time_steps must be set before calling time_response when tracing.")

        period = jnp.max(2.0 * jnp.pi / f_omega)
        periods_to_retain = const.N_PERIODS_TO_RETAIN
        T = period * periods_to_retain
        t_ss = jnp.max(self.model.t_steady_state(f_omega * 2.0 * jnp.pi, ss_tol=self.rtol)) * const.SAFETY_FACTOR_T_STEADY_STATE
        t0 = 0.0
        t1 = t_ss + T

        if kwargs.get("only_save_steady_state"):
            ts = jnp.linspace(t_ss, t1, self.n_time_steps * 10)
        else:
            n_periods = (t1 - t0) / T
            if self._is_tracer(n_periods):
                n_time_steps_total = self.n_time_steps * periods_to_retain
            else:
                n_time_steps_total = self.n_time_steps * int(math.ceil(float(n_periods)))
            ts = jnp.linspace(t0, t1, n_time_steps_total)

        max_steps_budget = self._max_steps_budget(t1 - t0, period)

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=max_steps_budget,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(f_amp, f_omega),
        )

        return sol.ts, sol.ys
        
    # TO DO: Add phase difference results
    def frequency_sweep(self,
             excitor: AbstractExcitation,
             sweeper: AbstractSweep,
            ) -> FrequencySweepResult:
        
        periods_to_retain = const.N_PERIODS_TO_RETAIN

        @filter_jit
        def solve_one_case(f_omega, f_amp, x0, v0):  
            x0 = jnp.full((self.model.n_modes,), x0)         
            v0 = jnp.full((self.model.n_modes,), v0)
            y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

            period = jnp.max(2.0 * jnp.pi / f_omega)
            T = period * periods_to_retain
            t_ss = jnp.max(self.model.t_steady_state(f_omega, ss_tol=self.rtol)) * const.SAFETY_FACTOR_T_STEADY_STATE
            
            t0 = 0.0
            t1 = t_ss + T
            ts = jnp.linspace(t_ss, t1, self.n_time_steps * periods_to_retain)
            
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=max_steps_budget,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(f_amp, f_omega),
            )

            # Treat any non-finite trajectories as failures to avoid polluting sweeps
            is_finite = jnp.all(jnp.isfinite(sol.ys))
            successful = jnp.logical_and(
                sol.result == diffrax.RESULTS.successful,
                is_finite,
            )

            xs = sol.ys[:, :self.model.n_modes]
            vs = sol.ys[:, self.model.n_modes:]

            max_x_total = jnp.where(
                successful,
                jnp.max(jnp.abs(jnp.sum(xs, axis=-1))),
                jnp.nan,
            )
            max_v_total = jnp.where(
                successful,
                jnp.max(jnp.abs(jnp.sum(vs, axis=-1))),
                jnp.nan,
            )

            max_x_modes = jnp.where(
                successful,
                jnp.max(jnp.abs(xs), axis=0),
                jnp.full_like(xs[0], jnp.nan),
            )
            max_v_modes = jnp.where(
                successful,
                jnp.max(jnp.abs(vs), axis=0),
                jnp.full_like(vs[0], jnp.nan),
            )

            return dict(
                f_omega=f_omega, 
                f_amp=f_amp,
                x0=x0, 
                v0=v0, 
                max_x_total=max_x_total, 
                max_v_total=max_v_total, 
                max_x_modes=max_x_modes, 
                max_v_modes=max_v_modes, 
                successful=successful
            )
            
        f_omegas = excitor.f_omegas
        f_amps = excitor.f_amps
        
        # TO DO: Check if this is appropriate
        if self.n_time_steps is None:
            if self._is_tracer(f_omegas):
                raise ValueError("n_time_steps must be set before calling frequency_sweep when tracing.")
            rtol = 0.01
            max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * jnp.max(f_omegas)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component

            n_time_steps = int(math.ceil(float(one_period * sampling_frequency)))
            self.n_time_steps = n_time_steps
        
        f_omegas, f_amps, x0s, v0s, shape = self.multistarter.generate_simulation_grid(self.model, f_omegas, f_amps)
        longest_period = jnp.max(2.0 * jnp.pi / f_omegas)
        t_ss_estimate = jnp.max(self.model.t_steady_state(f_omegas, ss_tol=self.rtol) * const.SAFETY_FACTOR_T_STEADY_STATE)
        t_span_estimate = t_ss_estimate + longest_period * periods_to_retain
        max_steps_budget = self._max_steps_budget(t_span_estimate, longest_period)

        flat_solutions = jax.vmap(solve_one_case, in_axes=(0, 0, 0, 0))(f_omegas, f_amps, x0s, v0s)

        periodic_solutions = jax.tree_util.tree_map(
            lambda leaf: leaf.reshape(shape[:-1] + leaf.shape[1:]),
            flat_solutions
        )

        successful_mask = periodic_solutions["successful"]
        n_successful = jnp.count_nonzero(successful_mask)
        n_total = successful_mask.size
        success_rate = jnp.where(n_total > 0, n_successful / n_total, 0.0)
        if not self._is_tracer(n_successful):
            n_successful = int(n_successful)
        if not self._is_tracer(success_rate):
            success_rate = float(success_rate)
        
        sweeped_periodic_solutions = sweeper.sweep(periodic_solutions)
                
        result = FrequencySweepResult(
            f_omegas=excitor.f_omegas,
            f_amps=excitor.f_amps,
            modal_forces=excitor.modal_forces,
            Q=self.model.Q,
            omega_0=self.model.omega_0,
            alpha=self.model.alpha,
            gamma=self.model.gamma,
            periodic_solutions=periodic_solutions,
            sweeped_periodic_solutions=sweeped_periodic_solutions,
            n_successful=n_successful,
            n_total=n_total,
            success_rate=success_rate,
        )

        return result

    @filter_jit
    def _rhs(self, t, y, args):
        f_amp, f_omega = args

        dy_dt = self.model.rhs(t, y, (f_amp, f_omega))

        return dy_dt
