import jax
import jax.numpy as jnp
from equinox import filter_jit
import diffrax

from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from .multistart.abstract_multistart import AbstractMultistart
from .multistart.linear_response_multistart import LinearResponseMultistart
from .. import constants as const 

class FixedTimeSolver(AbstractSolver):
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
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_disp: jax.Array,  
                 init_vel: jax.Array
                ):
        
        T = jnp.max(2.0 * jnp.pi / drive_freq)
        t_ss = jnp.max(self.model.t_steady_state(drive_freq * 2.0 * jnp.pi, ss_tol=self.rtol))

        t0 = 0.0
        t1 = t_ss + T

        ts = jnp.linspace(t_ss, t1, self.n_time_steps)
        y0 = jnp.concatenate([jnp.atleast_1d(init_disp), jnp.atleast_1d(init_vel)], axis=-1)

        sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Kvaerno5(),
                t0=t0, t1=t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(drive_amp, drive_freq),
            )

        return sol.ts, sol.ys
        
    def frequency_sweep(self,
             f_omega: jax.Array,
             f_amp: jax.Array, 
             sweep_direction: const.SweepDirection,
            ):
        pass

    @filter_jit
    def _rhs(self, t, y, args):
        f_amp, f_omega  = args

        dtau_dt = 2.0 * jnp.pi * (self.model.omega_ref / f_omega)
        tau = dtau_dt * t

        dy_dt = self.model.f(tau, y, args) * dtau_dt

        return dy_dt