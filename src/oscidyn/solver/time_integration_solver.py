import jax
import jax.numpy as jnp
from equinox import filter_jit
import diffrax

from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from .multistart.abstract_multistart import AbstractMultistart
from .multistart.linear_response_multistart import LinearResponseMultistart
from .. import constants as const 

class TimeIntegrationSolver(AbstractSolver):
    def __init__(self, multistart: AbstractMultistart = LinearResponseMultistart(),
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, 
                 verbose: bool = False, throw: bool = False):
        
        self.max_steps = max_steps
        self.n_time_steps = n_time_steps
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.throw = throw

        self.model: AbstractModel = None

    def time_response(self,
                 f_omega: jax.Array,  
                 f_amp: jax.Array, 
                 x0: jax.Array,  
                 v0: jax.Array,
                 **kwargs
                ):

        y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

        if self.n_time_steps is None:
            rtol = 0.01
            max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * f_omega.item()
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component
            
            n_time_steps = int(jnp.ceil(one_period * sampling_frequency))
            self.n_time_steps = n_time_steps

        T = jnp.max(2.0 * jnp.pi / f_omega) * 5
        t_ss = jnp.max(self.model.t_steady_state(f_omega * 2.0 * jnp.pi, ss_tol=self.rtol)) * 3
        t0 = 0.0
        t1 = t_ss + T

        if kwargs.get("only_save_steady_state"):
            ts = jnp.linspace(t_ss, t1, self.n_time_steps)
        else:
            n_periods = (t1 - t0) / T
            ts = jnp.linspace(t0, t1, self.n_time_steps * int(jnp.ceil(n_periods)))

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Kvaerno5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(f_amp, f_omega),
        )

        return sol.ts, sol.ys
        
    def frequency_sweep(self,
             f_omegas: jax.Array,
             f_amps: jax.Array, 
             sweep_direction: const.SweepDirection,
            ):
        
        @filter_jit
        def solve_one_case(f_omega, f_amp, x0, v0):  
            x0 = jnp.full((self.model.n_modes,), x0)         
            v0 = jnp.full((self.model.n_modes,), v0)
            y0 = jnp.concatenate([jnp.atleast_1d(x0), jnp.atleast_1d(v0)], axis=-1)

            T = jnp.max(2.0 * jnp.pi / f_omega) * const.N_PERIODS_TO_RETAIN
            t_ss = self.model.t_steady_state(f_omega, ss_tol=self.rtol) * const.SAFETY_FACTOR_T_STEADY_STATE
            
            t0 = 0.0
            t1 = t_ss + T
            ts = jnp.linspace(t_ss, t1, self.n_time_steps)
            
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(f_amp, f_omega),
            )
            return sol.ys
        
        # TO DO: Check if this is appropriate
        if self.n_time_steps is None:
            rtol = 0.01
            max_frequency_component = const.MAXIMUM_ORDER_SUPERHARMONICS * jnp.max(f_omegas)
            
            one_period = 2.0 * jnp.pi / max_frequency_component
            sampling_frequency = jnp.pi / (jnp.sqrt(2 * rtol)) * max_frequency_component

            n_time_steps = jnp.ceil(one_period * sampling_frequency).astype(int)
            self.n_time_steps = n_time_steps
        
        # TO DO: Determine the max amplitude based on the model or a fixed value
        max_abs_x0 = 1.0 * self.multistart.linear_response_factor
        x0 = jnp.linspace(-max_abs_x0, max_abs_x0, self.multistart.init_cond_shape[0])

        # TO DO: Determine the max velocity based on the model or a fixed value
        max_abs_v0 = 1.0 * self.multistart.linear_response_factor
        v0 = jnp.linspace(-max_abs_v0, max_abs_v0, self.multistart.init_cond_shape[1])

        f_omega_mesh, f_amp_mesh, x0_mesh, v0_mesh = jnp.meshgrid(f_omegas, f_amps, x0, v0, indexing="ij")

        f_omega_flat = f_omega_mesh.ravel()
        f_amp_flat = f_amp_mesh.ravel()
        x0_flat = x0_mesh.ravel()
        v0_flat = v0_mesh.ravel()
        
        ys = jax.vmap(solve_one_case)(f_omega_flat, f_amp_flat, x0_flat, v0_flat).reshape(f_omega_mesh.shape + (self.n_time_steps, self.model.n_states))
        
        xs = ys[..., :self.model.n_modes]
        vs = ys[..., self.model.n_modes:]

        max_x_total = jnp.max(jnp.abs(jnp.sum(xs, axis=-1)), axis=-1)
        max_v_total  = jnp.max(jnp.abs(jnp.sum(vs,  axis=-1)), axis=-1) 

        max_x_modes = jnp.max(jnp.abs(xs), axis=4)
        max_v_modes  = jnp.max(jnp.abs(vs),  axis=4) 
        
        return max_x_total

    @filter_jit
    def _rhs(self, t, y, args):
        f_amp, f_omega  = args

        dy_dt = self.model.f(t, y, args) 

        return dy_dt