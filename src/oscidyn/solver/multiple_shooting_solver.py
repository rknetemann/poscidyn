import jax
import jax.numpy as jnp
import diffrax
from functools import partial
from equinox import filter_jit

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
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096,
                 max_shooting_iterations: int = 20, shooting_tolerance: float = 1e-10,
                 m_segments: int = 20, max_branches: int = 5, multistart: AbstractMultistart = None,
                 rtol: float = 1e-4, atol: float = 1e-7, progress_bar: bool = False):

        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.max_shooting_iterations = max_shooting_iterations
        self.shooting_tolerance = shooting_tolerance
        self.m_segments = m_segments
        self.max_branches = max_branches
        self.progress_bar = progress_bar
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol

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
                       
        y0, y_max_displacement = self._calc_periodic_solution(drive_freq, drive_amp, init_disp, init_vel)

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



        # Solve shooting for each coarse combination to get y0

        y0, y_max = jax.vmap(self._calc_periodic_solution)(
            drive_freq_mesh_flat, drive_amp_mesh_flat, init_disp_mesh_flat, init_vel_mesh_flat
        )  # y0: (n_sim, n)  y_max: (n_sim, n_modes)

        # Plot the results
        plot_branch_exploration(drive_freq_mesh, drive_amp_mesh, y_max)

        return None

    @filter_jit
    def _rhs(self, tau, y, args):
        _drive_amp, drive_freq  = args

        t = tau / drive_freq
        dydt = self.model.f(t, y, args) / drive_freq

        return dydt
    
    @filter_jit
    def _aug_rhs(self, tau, y_aug, args):
        _drive_amp, drive_freq = args
        
        y  = y_aug[:self.n]
        y_max = y_aug[self.n:self.n * 2]
        X  = y_aug[self.n * 2:].reshape(self.n, self.n)

        t   = tau / drive_freq
        f   = self.model.f(t, y, args)
        f_y = self.model.f_y(t, y, args)

        dydt  = f / drive_freq
        dXdt  = ((f_y @ X) / drive_freq).reshape(-1)
        return jnp.hstack([dydt, y_max, dXdt])

    def _calc_periodic_solution(self,
                 driving_frequency: jax.Array, 
                 driving_amplitude: jax.Array,  
                 initial_displacement: jax.Array,
                 initial_velocity: jax.Array
                ):     

        ts=jnp.linspace(0.0, self.T, self.m + 1)

        initial_condition = jnp.array([initial_displacement, initial_velocity])

        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs),
            solver=diffrax.Tsit5(),
            t0=0.0, t1=self.T, dt0=None,
            max_steps=self.max_steps,
            y0=initial_condition,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_amplitude, driving_frequency),
        )

        s0 = sol.ys[:-1]

        eye_N = jnp.eye(self.n, dtype=initial_condition.dtype)
            
        @filter_jit
        def _integrate_segment(sk, t0k, t1k):
            X0 = eye_N.reshape(-1)
            y_max0 = sk
            sk0_aug = jnp.hstack([sk, y_max0, X0], dtype=initial_condition.dtype)  # Augmented state: [y; vec(X)]

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._aug_rhs),
                solver=diffrax.Tsit5(),
                t0=t0k, t1=t1k, dt0=None,
                max_steps=self.max_steps,
                y0=sk0_aug,
                saveat=diffrax.SaveAt(t1=True),
                throw=False,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),
            )

            sk_aug = sol.ys

            yk = sk_aug[-1, :self.n]
            y_max = sk_aug[-1, self.n:self.n * 2]
            XTk = sk_aug[-1, 2 * self.n:].reshape(self.n, self.n)
            Gk = XTk

            return sk, yk, y_max, XTk, Gk

        def _shooting_converged_cond(carry):
            k, done, y0, yT, y_max, XT, r = carry
            return jnp.logical_and(~done, k < self.max_shooting_iterations)

        def _shooting_iteration(carry):
            k, done, s, yT, y_max, XT, r = carry

            t0s = ts[:-1] # (m,)
            t1s = ts[1:] # (m,)

            sk, yk, y_max, XTk, Gk = jax.vmap(_integrate_segment, in_axes=(0, 0, 0))(s, t0s, t1s)
            # s: (m,n), s_end: (m,n), XTk: (m,2,2) TO DO: Generalize to n>1 DOF

            # Continuity defects F_j = y_{j+1} - s_{j+1}, j=0..m-2 (7.3.5.3):
            F_cont = yk[:-1] - s[1:] # (m-1, n)
            # Periodicity defect F_m = y_1^T(s_m) - s_1 (7.3.5.3):
            F_last = yk[-1] - s[0] # (n,)

            # Jacobian DF(s) has block-bidiagonal structure (7.3.5.5):
            #  [ G1  -I  0   ... ]
            #  [ 0   G2 -I   ... ]
            #  ...
            # We only need the condensed form (7.3.5.10):
            # (A + B G_{m-1} ... G1) Δs1 = w
            #
            # In periodic case, r(s1, sm) = s1 - sm
            A = -eye_N
            B = Gk[-1]

            # P = G_{m-2} ... G_0  (propagate Δs0 to Δs_{m-1})
            P = eye_N
            for i in range(self.m-1):          # i = 0..m-2
                P = Gk[i] @ P

            # S = Σ_{j=0}^{m-2} (G_{m-2} ... G_{j+1}) F_j   (suffix products)
            # build suffix products Q_j = Π_{i=j+1}^{m-2} G_i
            Q = jnp.zeros((self.m-1, self.n, self.n), dtype=initial_condition.dtype)
            Q = Q.at[self.m-2].set(eye_N)      # Q_{m-2} = I (empty product)

            def fill_suffix(i, Qacc):
                # i = 0..(m-3) -> j = (m-3 - i) runs downwards
                j = (self.m - 3) - i
                Qacc = Qacc.at[j].set(Gk[j+1] @ Qacc[j+1])
                return Qacc

            Q = jax.lax.fori_loop(0, self.m-2, fill_suffix, Q) if self.m > 2 else Q

            # sum_j Q_j @ F_cont[j]
            S_terms = jax.vmap(lambda Qj, Fj: Qj @ Fj)(Q, F_cont)  # (m-1, n)
            S       = jnp.sum(S_terms, axis=0)                    # (n,)

            # Condensed system:
            # (A + B P) Δs0 = -F_last - B S
            Jc = A + B @ P
            rhs = -(F_last + B @ S)

            # Solve for Δs0
            Δs0 = jnp.linalg.solve(Jc, rhs)

            # Propagate corrections:
            # Δs_{j+1} = G_j Δs_j + F_j, j=0..m-2
            Δs = jnp.zeros_like(s)            # (m, n)
            Δs = Δs.at[0].set(Δs0)

            def prop(i, ds):
                ds = ds.at[i+1].set(Gk[i] @ ds[i] + F_cont[i])
                return ds

            Δs = jax.lax.fori_loop(0, self.m-1, prop, Δs) if self.m > 1 else Δs

            # Update all segment starts
            s_new = s + Δs

            # Residual norm (max over segments)
            defects = jnp.concatenate([F_cont, F_last[None, :]], axis=0)  # (m, n)
            r_new   = jnp.max(jnp.linalg.norm(defects, axis=1))

            sh_done = r_new < self.shooting_tolerance
            k += 1

            # keep yT, XT shapes consistent with init; they are not used downstream
            return k, sh_done, s_new, yk[-1], y_max, XT, r_new          
        
        init_carry = (
            jnp.array(0, dtype=jnp.int32), # k
            jnp.array(False), # done
            s0, # s0: (m,n)
            jnp.full((self.n,), jnp.inf, dtype=initial_condition.dtype), # yT: (n,)
            s0, # y_max: (n,n)
            jnp.eye(self.n, dtype=initial_condition.dtype), # XT: (n,n)
            jnp.array(jnp.inf, dtype=initial_condition.dtype), # r: scalar float
        )

        k, ls_done, s_final, yT, y_max, XT, r = jax.lax.while_loop(
            _shooting_converged_cond, _shooting_iteration, init_carry
        )
        max_displacement = jnp.max(jnp.abs(y_max[:, :self.model.n_modes]), axis=0)  # (n_modes,)

        y0 = s_final[0]                              # (n,)
        y0 = jnp.where(ls_done, y0, jnp.nan * y0)   # keep shape
        max_displacement = jnp.where(ls_done, max_displacement, jnp.nan * max_displacement)

        return y0, max_displacement  # Return time and state (displacement and velocity)

    def _solve(self,
               rhs: callable,
               wf: bool,
               t0: float,
               t1: float,
               y0: jax.Array,
               driving_frequency: float,
               driving_amplitude: jax.Array) -> diffrax.Solution:
        
        ts = jnp.linspace(t0, t1, self.n_time_steps)

        T = 2.0 * jnp.pi
        dtmax = T / const.DT_MAX_FACTOR / self.m_segments
        dtmin = T / const.DT_MIN_FACTOR
        max_steps = self.max_steps

        def _saveat_fn(t, y_aug, args):
            y_current = y_aug[:self.n]
            y_max = y_aug[self.n:self.n * 2]
            y_max = jnp.maximum(y_max, y_current)

            return y_aug.at[self.n:self.n * 2].set(y_max)
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(rhs),
            solver=diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=max_steps,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts) if wf else diffrax.SaveAt(t1=True),
            throw=False,
            progress_meter=diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol
            ),
            args=(driving_amplitude, driving_frequency),
        )
        return sol

