import math
import warnings
import jax
from jaxtyping import Array, PyTree, Float
import jax.numpy as jnp
from equinox import filter_jit
import diffrax
from jax import core as jax_core
from typing import Optional

from .abstract_solver import AbstractSolver
from ..oscillator.abstract_oscillator import AbstractOscillator
from ..multistart.abstract_multistart import AbstractMultistart
from ..multistart.linear_response import LinearResponse
from ..excitation.abstract_excitation import AbstractExcitation
from ..excitation.abstract_periodic_excitation import AbstractPeriodicExcitation
from ..excitation.free_vibration import FreeVibration
from ..synthetic_sweep.abstract_synthetic_sweep import AbstractSyntheticSweep
from ..synthetic_sweep.nearest_neighbour import NearestNeighbour
from ..response_measure.abstract_response_measure import AbstractResponseMeasure
from ..result.frequency_sweep import FrequencySweep, ResponseData, DemodulationResult, ScalarResponseResult, BranchResult
from ..response_measure.demodulation import Demodulation
from ..response_measure.rms import RMS
from ..response_measure.min import Min
from ..response_measure.max import Max
from ..result.time_response import TimeResponse

class TimeIntegration(AbstractSolver):
    def __init__(self, oscillator: AbstractOscillator, excitation: AbstractExcitation = FreeVibration(), 
                 response_measure: AbstractResponseMeasure = None,
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = 8, max_steps: int = 4096, 
                 multistart: AbstractMultistart = LinearResponse(), synthetic_sweep: AbstractSyntheticSweep = NearestNeighbour(),
                 t_steady_state_factor: float = 1.2, periods_to_retain: int = 4, max_order_superharmonics: int = 3,
                 verbose: bool = False, throw: bool = False):

        super().__init__(oscillator, excitation, response_measure)
        
        self.max_steps = max_steps
        self.n_time_steps = n_time_steps
        self.rtol = rtol
        self.atol = atol
        self.t_steady_state_factor = t_steady_state_factor
        self.periods_to_retain = periods_to_retain
        self.max_order_superharmonics = max_order_superharmonics
        self.verbose = verbose
        self.throw = throw

        self.multistart = multistart
        self.synthetic_sweep = synthetic_sweep

    @staticmethod
    def _is_tracer(value) -> bool:
        """Check whether a value is being traced by JAX."""
        return isinstance(value, jax_core.Tracer)

    def time_response(self,
            x0: jax.Array,  
            v0: jax.Array,
            *,
            t: Optional[Float] = None,
            **kwargs,
        ):
        """Compute the time response for one excitation frequency.

        Args:
            x0: Initial displacement.
            v0: Initial velocity.
            t: Time to compute the response for. If not provided, the time will be computed based on the excitation frequency and the number of periods to retain.
            **kwargs: Additional solver options, such as
                ``only_save_steady_state``.
        """
        is_periodic = isinstance(self.excitation, AbstractPeriodicExcitation)
        omega = None
        if is_periodic:
            if self.excitation.omega is None:
                raise ValueError(
                    "Time response requires an omega for periodic excitations."
                )
            omega = jnp.asarray(self.excitation.omega)
            if omega.size != 1:
                raise ValueError(
                    "Time response requires exactly one excitation frequency."
                )
            if not self._is_tracer(omega) and bool(jnp.any(omega <= 0)):
                raise ValueError("omega must contain only positive frequencies.")
            args = {"omega": omega, "lambda": self.excitation.lambdas}
        else:
            args = {"lambda": self.excitation.lambdas}

        if t is not None:
            t = jnp.asarray(t)
            if t.ndim != 0:
                raise ValueError("t must be a positive scalar duration.")
            if not self._is_tracer(t) and bool(t <= 0):
                raise ValueError("t must be a positive scalar duration.")
            if kwargs.get("only_save_steady_state"):
                raise ValueError(
                    "only_save_steady_state cannot be used when t is specified."
                )
            if self.n_time_steps is None:
                raise ValueError(
                    "n_time_steps must be set when t is specified."
                )
            t0 = 0.0
            t1 = t
            ts = jnp.linspace(t0, t1, self.n_time_steps)
        else:
            if not is_periodic:
                raise ValueError(
                    "Time response requires t for a non-periodic excitation."
                )

            if self.n_time_steps is None:
                if self._is_tracer(omega):
                    raise ValueError(
                        "n_time_steps must be set before calling time_response when tracing."
                    )
                resolution_rtol = 0.01
                max_frequency_component = self.max_order_superharmonics * jnp.max(omega)
                one_period = 2.0 * jnp.pi / max_frequency_component
                sampling_frequency = (
                    jnp.pi / jnp.sqrt(2 * resolution_rtol) * max_frequency_component
                )
                self.n_time_steps = int(
                    math.ceil(float(one_period * sampling_frequency))
                )

            period = jnp.max(2.0 * jnp.pi / omega)
            retained_duration = period * self.periods_to_retain
            steady_state_time = (
                jnp.max(self.oscillator.t_steady_state(omega, ss_tol=self.rtol))
                * self.t_steady_state_factor
            )
            t0 = 0.0
            t1 = steady_state_time + retained_duration

            if kwargs.get("only_save_steady_state"):
                ts = jnp.linspace(steady_state_time, t1, self.n_time_steps)
            else:
                n_periods = (t1 - t0) / retained_duration
                if self._is_tracer(n_periods):
                    n_time_steps_total = self.n_time_steps * self.periods_to_retain
                else:
                    n_time_steps_total = self.n_time_steps * int(
                        math.ceil(float(n_periods))
                    )
                ts = jnp.linspace(t0, t1, n_time_steps_total)

        y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=args,
        )

        return TimeResponse(
            time=sol.ts,
            displacement=sol.ys[:, :self.oscillator.n_dof],
            velocity=sol.ys[:, self.oscillator.n_dof:],
        )
        
    def frequency_sweep(self, omegas: Array) -> FrequencySweep:
        self._validate_frequency_sweep()
        sweep_frequencies = jnp.asarray(omegas)

        @filter_jit
        def solve_one_case(omega, x0, v0):
            x0 = jnp.full((self.oscillator.n_modes,), x0)         
            v0 = jnp.full((self.oscillator.n_modes,), v0)
            y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

            period = jnp.max(2.0 * jnp.pi / omega)
            T = period * self.periods_to_retain
            t_ss = jnp.max(self.oscillator.t_steady_state(omega, ss_tol=self.rtol)) * self.t_steady_state_factor
            
            t0 = 0.0
            t1 = t_ss + T

            ts = jnp.linspace(
                t_ss,
                t1,
                self.n_time_steps * self.periods_to_retain,
                endpoint=False,
            )

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args={"omega": omega},
            )

            # Treat any non-finite trajectories as failures to avoid polluting sweeps
            is_finite = jnp.all(jnp.isfinite(sol.ys))
            successful = jnp.logical_and(
                sol.result == diffrax.RESULTS.successful,
                is_finite,
            )

            xs = sol.ys[:, :self.oscillator.n_dof]
            vs = sol.ys[:, self.oscillator.n_dof:]

            response = self.response_measure(
                xs=xs,
                ts=ts,
                drive_omega=omega,
            )
            if not isinstance(response, dict):
                raise ValueError(
                    "response_measure must return a dict with 'modal' and 'total' blocks."
                )

            modal = response["modal"]
            total = response["total"]

            modal_amplitude = modal["amplitude"]
            modal_phase = modal["phase"]
            modal_response_frequency = modal.get("response_frequency")
            if modal_response_frequency is None:
                modal_response_frequency = jnp.full_like(modal_phase, jnp.nan)

            total_amplitude = total["amplitude"]
            total_phase = total["phase"]
            total_response_frequency = total.get("response_frequency")
            if total_response_frequency is None:
                total_response_frequency = jnp.full_like(total_phase, jnp.nan)

            return dict(
                omega=omega, 
                x0=x0, 
                v0=v0,
                modal_amplitude=modal_amplitude,
                modal_phase=modal_phase,
                modal_response_frequency=modal_response_frequency,
                total_amplitude=total_amplitude,
                total_phase=total_phase,
                total_response_frequency=total_response_frequency,
                successful=successful
            )
            
        # TO DO: Check if this is appropriate
        if self.n_time_steps is None:
            if self._is_tracer(omegas):
                raise ValueError("n_time_steps must be set before calling frequency_sweep when tracing. ")
            rtol = 0.01
            max_frequency_component = self.max_order_superharmonics * jnp.max(omegas)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component

            n_time_steps = int(math.ceil(float(one_period * sampling_frequency)))
            self.n_time_steps = n_time_steps
        
        omegas, x0s, v0s, shape = self.multistart.generate_simulation_grid(self.oscillator, omegas)
        longest_period = jnp.max(2.0 * jnp.pi / omegas)
        t_ss_estimate = jnp.max(self.oscillator.t_steady_state(omegas, ss_tol=self.rtol) * self.t_steady_state_factor)
        t_span_estimate = t_ss_estimate + longest_period * self.periods_to_retain

        flat_solutions = jax.vmap(solve_one_case, in_axes=(0, 0, 0))(omegas, x0s, v0s)

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
        
        sweeped_periodic_solutions = self.synthetic_sweep.sweep(periodic_solutions)
                
        def make_response(direction, total=False):
            suffix = "_total" if total else ""
            value = sweeped_periodic_solutions.get(direction + suffix)
            if isinstance(self.response_measure, Demodulation):
                return DemodulationResult(
                    value, sweeped_periodic_solutions.get(direction + suffix + "_phase"),
                    sweeped_periodic_solutions.get(direction + suffix + "_demod_freq"))
            measure = "rms" if isinstance(self.response_measure, RMS) else "minimum" if isinstance(self.response_measure, Min) else "maximum" if isinstance(self.response_measure, Max) else "value"
            return ScalarResponseResult(value, measure)
        forward = BranchResult(make_response("forward"), make_response("forward", True))
        backward = BranchResult(make_response("backward"), make_response("backward", True))
        result = FrequencySweep(
            frequency=sweep_frequencies,
            forward=forward,
            backward=backward,
            stats={
                "n_successful": n_successful,
                "n_total": n_total,
                "success_rate": success_rate,
            },
        )
        return result

    @filter_jit
    def _rhs(self, t, y, args, **kwargs):
        q, dq_dt   = jnp.split(y, 2)

        dy_dt = jnp.concatenate([dq_dt,  self.oscillator.f_i(t, y, args, **kwargs) - self.excitation.f_e(t, y, args, **kwargs)], axis=0)
        return dy_dt
