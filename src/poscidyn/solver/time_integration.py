import math
import jax
import jax.numpy as jnp
from equinox import filter_jit
import diffrax
from jax import core as jax_core

from .abstract_solver import AbstractSolver
from ..oscillator.abstract_oscillator import AbstractOscillator
from ..multistart.abstract_multistart import AbstractMultistart
from ..multistart.linear_response import LinearResponse
from ..excitation.abstract_excitation import AbstractExcitation
from ..synthetic_sweep.abstract_synthetic_sweep import AbstractSyntheticSweep
from ..synthetic_sweep.nearest_neighbour import NearestNeighbour
from ..response_measure.abstract_response_measure import AbstractResponseMeasure
from ..result.frequency_sweep import FrequencySweep, Phasors

class TimeIntegration(AbstractSolver):
    def __init__(self, rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, 
                 multistart: AbstractMultistart = LinearResponse(), synthetic_sweep: AbstractSyntheticSweep = NearestNeighbour(),
                 t_steady_state_factor: float = 1.2, periods_to_retain: int = 4, max_order_superharmonics: int = 3,
                 verbose: bool = False, throw: bool = False):
        
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

        self.oscillator: AbstractOscillator = None
        self.excitation: AbstractExcitation = None
        self.response_measure: AbstractResponseMeasure = None

    @staticmethod
    def _is_tracer(value) -> bool:
        """Check whether a value is being traced by JAX (e.g. inside vmap/jit)."""
        return isinstance(value, jax_core.Tracer)

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
            max_frequency_component = self.max_order_superharmonics * jnp.max(f_omega)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component
            
            n_time_steps = int(math.ceil(float(one_period * sampling_frequency)))
            self.n_time_steps = n_time_steps
        elif self.n_time_steps is None:
            raise ValueError("n_time_steps must be set before calling time_response when tracing.")

        period = jnp.max(2.0 * jnp.pi / f_omega)
        periods_to_retain = self.periods_to_retain
        T = period * periods_to_retain
        t_ss = jnp.max(self.oscillator.t_steady_state(f_omega * 2.0 * jnp.pi, ss_tol=self.rtol)) * self.t_steady_state_factor
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

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args={"f_amp": f_amp, "f_omega": f_omega},
        )

        return sol.ts, sol.ys
        
    def frequency_sweep(self) -> FrequencySweep:
        @filter_jit
        def solve_one_case(f_omega, f_amp, x0, v0):  
            x0 = jnp.full((self.oscillator.n_modes,), x0)         
            v0 = jnp.full((self.oscillator.n_modes,), v0)
            y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

            period = jnp.max(2.0 * jnp.pi / f_omega)
            T = period * self.periods_to_retain
            t_ss = jnp.max(self.oscillator.t_steady_state(f_omega, ss_tol=self.rtol)) * self.t_steady_state_factor
            
            t0 = 0.0
            t1 = t_ss + T
            ts = jnp.linspace(t_ss, t1, self.n_time_steps * self.periods_to_retain)

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args={"f_amp": f_amp, "f_omega": f_omega},
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
                drive_omega=f_omega,
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
                f_omega=f_omega, 
                f_amp=f_amp,
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
            
        f_omegas = self.excitation.f_omegas
        f_amps = self.excitation.f_amps
        
        # TO DO: Check if this is appropriate
        if self.n_time_steps is None:
            if self._is_tracer(f_omegas):
                raise ValueError("n_time_steps must be set before calling frequency_sweep when tracing.")
            rtol = 0.01
            max_frequency_component = self.max_order_superharmonics * jnp.max(f_omegas)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component

            n_time_steps = int(math.ceil(float(one_period * sampling_frequency)))
            self.n_time_steps = n_time_steps
        
        f_omegas, f_amps, x0s, v0s, shape = self.multistart.generate_simulation_grid(self.oscillator, f_omegas, f_amps)
        longest_period = jnp.max(2.0 * jnp.pi / f_omegas)
        t_ss_estimate = jnp.max(self.oscillator.t_steady_state(f_omegas, ss_tol=self.rtol) * self.t_steady_state_factor)
        t_span_estimate = t_ss_estimate + longest_period * self.periods_to_retain

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
        
        sweeped_periodic_solutions = self.synthetic_sweep.sweep(periodic_solutions)
                
        modal_coordinates = Phasors(
            amplitudes={
                "forward": sweeped_periodic_solutions.get("forward"),
                "backward": sweeped_periodic_solutions.get("backward"),
            },
            phases={
                "forward": sweeped_periodic_solutions.get("forward_phase"),
                "backward": sweeped_periodic_solutions.get("backward_phase"),
            },
            demod_freqs={
                "forward": sweeped_periodic_solutions.get("forward_demod_freq"),
                "backward": sweeped_periodic_solutions.get("backward_demod_freq"),
            }
        )

        modal_superposition = Phasors(
            amplitudes={
                "forward": sweeped_periodic_solutions.get("forward_total"),
                "backward": sweeped_periodic_solutions.get("backward_total"),
            },
            phases={
                "forward": sweeped_periodic_solutions.get("forward_total_phase"),
                "backward": sweeped_periodic_solutions.get("backward_total_phase"),
            },
            demod_freqs={
                "forward": sweeped_periodic_solutions.get("forward_total_demod_freq"),
                "backward": sweeped_periodic_solutions.get("backward_total_demod_freq"),
            },
        )

        result = FrequencySweep(
            modal_coordinates=modal_coordinates,
            modal_superposition=modal_superposition,
            stats={
                "n_successful": n_successful,
                "n_total": n_total,
                "success_rate": success_rate,
            },
        )

        return result

    @filter_jit
    def _rhs(self, t, y, args):
        q, dq_dt   = jnp.split(y, 2)

        dy_dt = jnp.concatenate([dq_dt,  self.oscillator.f_i(t, y, args) - self.excitation.f_e(t, y, args)], axis=0)
        return dy_dt
