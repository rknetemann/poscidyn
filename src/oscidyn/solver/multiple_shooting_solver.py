import jax
import jax.numpy as jnp
import diffrax
from functools import partial
from equinox import filter_jit
import optimistix as optx


from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from .multistart.abstract_multistart import AbstractMultistart
from ..utils.plotting import plot_branch_exploration, plot_branch_selection
from .utils.coarse_grid import gen_coarse_grid_1, gen_grid_2
from .utils.branch_selection import select_branches
from .utils.uique_solutions import get_unique_solutions
from .. import constants as const 

import numpy as _np
import matplotlib.pyplot as _plt

class MultipleShootingSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = None, max_steps: int = 4096,
                 max_shooting_iterations: int = 20,
                 m_segments: int = 20, max_branches: int = 5, multistart: AbstractMultistart = None,
                 rtol: float = 1e-4, atol: float = 1e-7, progress_bar: bool = False,
                 feas_tol: float = 1e-5, step_tol: float = 1e-6,
                 ls_beta: float = 0.5, ls_c: float = 1e-4, ls_max_iters: int = 8):

        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.max_shooting_iterations = max_shooting_iterations
        self.m_segments = m_segments
        self.max_branches = max_branches
        self.progress_bar = progress_bar
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol
        self.feas_tol = feas_tol         # <<< NEW
        self.step_tol = step_tol         # <<< NEW
        self.ls_beta = ls_beta
        self.ls_c = ls_c
        self.ls_max_iters = ls_max_iters

        self.model: AbstractModel = None
        self.n = None
        self.m = None

        self.T = 2.0 * jnp.pi
        self.dtmax = self.T / const.DT_MAX_FACTOR / self.m_segments
        self.dtmin = self.T / const.DT_MIN_FACTOR
        self.t0 = 0.0
        self.t1 = self.T
   
    def time_response(self,
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_disp: jax.Array,  
                 init_vel: jax.Array
                ):
        
        self.n = self.model.n_modes * 2
        self.m = self.m_segments

        y0_guess = jnp.array([init_disp, init_vel]).flatten()

        y0, x_max = self._calc_periodic_solution(drive_freq, drive_amp, y0_guess)

        def _solve_one_period(y0):               
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=self.T,
                dt0=None,
                max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, self.T, self.n_time_steps)),
                throw=False,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(drive_amp, drive_freq),
            )
            return sol.ts, sol.ys

        ts, ys = jax.lax.cond(
            jnp.isnan(y0).any(),
            lambda: (jnp.zeros((self.n_time_steps,)),
                     jnp.zeros((self.n_time_steps, self.model.n_modes * 2))),
            lambda: _solve_one_period(y0)
        )

        return ts, ys
    
    def frequency_sweep(self,
             drive_freq: jax.Array,   # (1,) or scalar
             drive_amp: jax.Array,    # (n_modes,)
             sweep_direction: const.SweepDirection,
            ):
    
        self.n = self.model.n_modes * 2
        self.m = self.m_segments

        drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = gen_grid_2(
            self.model, drive_freq, drive_amp
        )
        drive_freq_mesh_flat = drive_freq_mesh.ravel()                       # (n_sim,)
        drive_amp_mesh_flat  = drive_amp_mesh.ravel()                        # (n_sim,)
        init_disp_mesh_flat  = init_disp_mesh.ravel()                        # (n_sim,)
        init_vel_mesh_flat   = init_vel_mesh.ravel()                         # (n_sim,)

        y0_guess = jnp.array([init_disp_mesh_flat, init_vel_mesh_flat]).reshape((-1, self.model.n_modes * 2))  # (n_sim, n_modes*2) 

        # Solve shooting for each coarse combination to get y0
        y0, x_max = jax.vmap(self._calc_periodic_solution)(
            drive_freq_mesh_flat, drive_amp_mesh_flat, y0_guess
        )  # y0: (n_sim, n_modes*2)  y_max: (n_sim, n_modes)

        # Plot the results
        #plot_branch_exploration(drive_freq_mesh, drive_amp_mesh, y_max, mu)

        return x_max
    
    @filter_jit
    def _rhs(self, tau, y, args):
        _drive_amp, drive_freq  = args

        t = tau / drive_freq
        dydt = self.model.f(t, y, args) / drive_freq

        return dydt

    @filter_jit
    def _calc_periodic_solution(self,
                                driving_frequency: jax.Array,
                                driving_amplitude: jax.Array,
                                y0_guess: jax.Array):

        m_segments = self.m_segments
        T = self.T
        ts = jnp.linspace(0.0, T, m_segments + 1)  # (n_modes * 2,)

        sol0 = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs), solver=diffrax.Tsit5(),
            t0=0.0, t1=T, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_amplitude, driving_frequency),
        )
        s0 = sol0.ys[:-1]  # (m_segments, n_modes * 2)

        @filter_jit
        def _integrate_segment(sk, t0k, t1k):
            solk = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=t0k, t1=t1k, dt0=None,
                y0=sk,
                adjoint=diffrax.RecursiveCheckpointAdjoint(),
                saveat=diffrax.SaveAt(t1=True),
                throw=False,
                max_steps=self.max_steps,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),
            )
            return solk.ys.squeeze(0)   # (n_modes * 2,)

        def _residual(s, _args=None):               # s: (m_segments, n_modes * 2)
            def _one(carry, i):
                sk  = s[i]
                sk1 = s[(i + 1) % m_segments]    # wrap for periodicity
                Phi = _integrate_segment(sk, ts[i], ts[i + 1])  # (n_modes * 2,)
                r = Phi - sk1
                return carry, r

            _, Rs = jax.lax.scan(_one, None, jnp.arange(m_segments))  # Rs: (m_segments, n_modes * 2)
            return Rs.reshape((-1,))                         # (m_segments * n_modes * 2,)

        solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-9) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
        sol = optx.least_squares(_residual, solver, y0=s0, options={"jac": "bwd"}, max_steps=self.max_shooting_iterations) 
        max_norm = optx.max_norm(_residual(sol.value))

        # This again is probably very sensitive to the initial conditions again, have to think about that
        def _integrate_periodic_solution(y0):
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs), solver=diffrax.Tsit5(),
                t0=0.0, t1=T, dt0=None,
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, T, self.n_time_steps)),
                throw=False,
                max_steps=self.max_steps,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),
            )

            xs  = sol.ys[:, :self.model.n_modes]  
            x_max_per_mode = jnp.max(jnp.abs(xs), axis=0)
            x_max = jnp.max(x_max_per_mode)           
            return y0, x_max
        
        y0, x_max = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda: _integrate_periodic_solution(sol.value[0]),
            lambda: (jnp.zeros((self.model.n_modes*2,)),
                     jnp.zeros(()))            
        )

        return y0, x_max
