import jax
import jax.numpy as jnp
import diffrax
from equinox import filter_jit, filter_checkpoint
import optimistix as optx
import lineax as lx
import jaxopt
import time

from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from .multistart.abstract_multistart import AbstractMultistart
from .multistart.linear_response_multistart import LinearResponseMultistart
from .. import constants as const 
from .utils.polynomials import LagrangeBasis

class CollocationSolver(AbstractSolver):
    def __init__(self,  max_iterations: int = 20, N_elements: int = 10, K_polynomial_degree: int = 2, multistart: AbstractMultistart = LinearResponseMultistart(),
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, 
                 verbose: bool = False, throw: bool = False):

        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.N_elements = N_elements
        self.K_polynomial_degree = K_polynomial_degree
        self.order_approx = 5
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose
        self.throw = throw

        self.model: AbstractModel = None

        self.multistart.verbose = self.verbose

        self.t0 = 0.0
        self.t1 = 1.0
        
        self.ts = jnp.linspace(self.t0, self.t1, self.N_elements * (self.K_polynomial_degree + 1))  # (N_elements*(K_polynomial_degree+1),)
        self.ts_segments = self.ts.reshape(self.N_elements, self.K_polynomial_degree + 1)  # (N_elements, K_polynomial_degree+1)
        self.he = (self.t1 - self.t0) / self.N_elements 
       
        self.tau = jnp.linspace(0, 1, self.K_polynomial_degree + 1)  # (K_polynomial_degree+1,)
        self.lagrange_basis = LagrangeBasis(self.tau)
        self.D = self.lagrange_basis.differentiation_matrix()  # (K_polynomial_degree+1, K_polynomial_degree+1)
        self.f = jax.vmap(self._rhs, in_axes=(0, 0, None))  # (K+1, n_modes*2)
   
    def time_response(self,
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_disp: jax.Array,  
                 init_vel: jax.Array
                ):

        y0_guess = jnp.concatenate([jnp.atleast_1d(init_disp), jnp.atleast_1d(init_vel)], axis=-1)
        Y0 = self._calc_Y0((drive_amp, drive_freq), y0_guess)
        y0, y_max = self._calc_periodic_solution((drive_amp, drive_freq), Y0)

        ts = jnp.linspace(self.t0, self.t1, self.n_time_steps)

        def _solve_one_period(y0):               
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Kvaerno5(),
                t0=self.t0, t1=self.t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                throw=self.throw,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(drive_amp, drive_freq),
            )
            return sol.ts, sol.ys

        _, ys = jax.lax.cond(
            jnp.isnan(y0).any(),
            lambda: (jnp.zeros((self.n_time_steps,)),
                     jnp.zeros((self.n_time_steps, self.model.n_modes * 2))),
            lambda: _solve_one_period(y0)
        )

        ts = ts * 1 / (drive_freq / (2 * jnp.pi))

        xs = ys[:, :self.model.n_modes]
        vs = ys[:, self.model.n_modes:] / drive_freq

        ys = jnp.concatenate([xs, vs], axis=-1)

        return ts, ys
    
    def frequency_sweep(self,
             f_omega: jax.Array,
             f_amp: jax.Array, 
             sweep_direction: const.SweepDirection,
            ):

        f_omega_mesh, f_amp_mesh, x0_mesh, v0_mesh = self.multistart.generate_simulation_grid(
            self.model, f_omega.flatten(), f_amp.flatten()
        )

        f_omega_mesh_flat = f_omega_mesh.ravel()
        f_amp_mesh_flat  = f_amp_mesh.ravel()
        y0_guess = jnp.stack([x0_mesh, v0_mesh], axis=-1).reshape(-1, self.model.n_states) 

        Y0 = jax.vmap(self._calc_Y0, in_axes=((0,0), 0))(
            (f_amp_mesh_flat, f_omega_mesh_flat), y0_guess
        )  # Y0: (n_sim, N_elements*(K_polynomial_degree+1), n_states)

        y0, y_max = jax.vmap(self._calc_periodic_solution, in_axes=((0,0), 0))(
            (f_amp_mesh_flat, f_omega_mesh_flat), Y0
        )  # y0: (n_sim, n_states)  y_max: (n_sim, n_modes)

        return {
            "y0": y0,
            "y_max": y_max,
        }
    
    @filter_jit
    def _calc_Y0(self,
            args: jax.Array,
            y0_guess: jax.Array):
                
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs), solver=diffrax.Kvaerno5(),
            t0=self.t0, t1=self.t1, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=self.ts),
            throw=self.throw,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=args,
        )

        def _process_success(sol):
            Y0 = sol.ys
            return Y0

        def _initial_guess_fallback():
            Y0 = jnp.tile(y0_guess, (self.N_elements * (self.K_polynomial_degree + 1), 1))
            return Y0

        is_successful = sol.result == diffrax.RESULTS.successful
        is_finite = jnp.all(jnp.isfinite(sol.ys))
        fallback = jnp.logical_not(jnp.logical_and(is_successful, is_finite))
        
        Y0 = jax.lax.cond(
            fallback,
            lambda: _initial_guess_fallback(),
            lambda: _process_success(sol)
        )

        return Y0
    
    @filter_jit
    def _calc_periodic_solution(self,
            args: jax.Array,
            Y0: jax.Array):
        
        solver = optx.LevenbergMarquardt(rtol=self.rtol, atol=self.atol, norm=optx.max_norm) 
        sol = optx.least_squares(self._residual, solver, args=args, y0=Y0, max_steps=self.max_iterations, throw=self.throw) 

        def _return_fail():
            y0_fail = jnp.full((self.model.n_modes * 2,), jnp.nan)
            y_max_fail = jnp.full((self.model.n_modes,), jnp.nan)
            return y0_fail, y_max_fail

        def _postprocess_periodic_solution(Y):    
            Y_series = Y.reshape(-1, self.model.n_modes * 2)
            y_max = jnp.max(jnp.abs(Y_series[:, 0::2]), axis=0)

            #self._analyze_periodic_solution(Y[0], args)

            y0 = Y[0]
            return y0, y_max
        
        is_successful = sol.result == optx.RESULTS.successful
        is_finite = jnp.all(jnp.isfinite(sol.value))
        y0, y_max = jax.lax.cond(
            jnp.logical_and(is_successful, is_finite),
            lambda: _postprocess_periodic_solution(sol.value),
            lambda: _return_fail()
        )
        return y0, y_max

    def _residual(self, Y, args):
        Y_segments = Y.reshape(self.N_elements, self.K_polynomial_degree + 1, -1)

        D = self.D  # (K+1, K+1)

        def _one_segment(ts_i, Y_i):
            # D @ Y_i uses (K+1, K+1) x (K+1, n) -> (K+1, n)
            dz_dtau = D @ Y_i
            dy_dt   = (1.0 / self.he) * dz_dtau

            dy_dt_eff = dy_dt[1:, :]            # K x n
            t_i_eff   = ts_i[1:]                # K
            r_col     = dy_dt_eff - self.f(t_i_eff, Y_i[1:, :], args)  # K x n
            return r_col

        _batch_segments = jax.vmap(_one_segment, in_axes=(0, 0))
        R_col  = _batch_segments(self.ts_segments, Y_segments)     # (N, K, n)
        R_cont = Y_segments[1:, 0, :] - Y_segments[:-1, -1, :]     # (N-1, n)
        R_per  = Y_segments[0, 0, :] - Y_segments[-1, -1, :]       # (n,)

        R = jnp.concatenate([R_col.flatten(), R_cont.flatten(), R_per.flatten()])
        return R
    
    def _analyze_periodic_solution(self):
        raise NotImplementedError

    @filter_jit
    def _rhs(self, t, y, args):
        f_amp, f_omega  = args

        dtau_dt = 2.0 * jnp.pi * (self.model.omega_ref / f_omega)
        tau = dtau_dt * t

        dy_dt = self.model.f(tau, y, args) * dtau_dt

        return dy_dt

