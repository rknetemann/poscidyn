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
                 max_shooting_iterations: int = 20,
                 m_segments: int = 20, max_branches: int = 5, multistart: AbstractMultistart = None,
                 rtol: float = 1e-4, atol: float = 1e-7, progress_bar: bool = False,
                 feas_tol: float = 1e-2, step_tol: float = 1e-3):

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
        y0, y_max, mu = jax.vmap(self._calc_periodic_solution)(
            drive_freq_mesh_flat, drive_amp_mesh_flat, init_disp_mesh_flat, init_vel_mesh_flat
        )  # y0: (n_sim, n_modes)  y_max: (n_sim, n_modes)

        # Plot the results
        #plot_branch_exploration(drive_freq_mesh, drive_amp_mesh, y_max, mu)

        return y0, y_max, mu

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


    @filter_jit
    def _max_save_fn(self, t, y_aug, args):
        n = self.n
        y     = y_aug[:n]
        y_max = y_aug[n:2*n]
        y_max = jnp.maximum(y_max, y)
        return y_aug.at[n:2*n].set(y_max)


    @filter_jit
    def _calc_periodic_solution(self,
                                driving_frequency: jax.Array,
                                driving_amplitude: jax.Array,
                                initial_displacement: jax.Array,
                                initial_velocity: jax.Array):

        n = self.model.n_modes * 2
        m = self.m_segments
        T = self.T
        ts = jnp.linspace(0.0, T, m + 1)

        y0_init = jnp.array([initial_displacement, initial_velocity])
        term = diffrax.ODETerm(self._rhs)
        solver = diffrax.Tsit5()

        sol0 = diffrax.diffeqsolve(
            term, solver,
            t0=0.0, t1=T, dt0=None,
            y0=y0_init,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_amplitude, driving_frequency),
        )
        s = sol0.ys[:-1]  # (m, n)

        eye_N = jnp.eye(n, dtype=y0_init.dtype)

        @filter_jit
        def _integrate_segment(sk, t0k, t1k):
            X0 = eye_N.reshape(-1)
            y_aug0 = jnp.hstack([sk, sk, X0]).astype(y0_init.dtype)  # [y, y_max, vec(X)]

            solk = diffrax.diffeqsolve(
                diffrax.ODETerm(self._aug_rhs),
                solver,
                t0=t0k, t1=t1k, dt0=None,
                y0=y_aug0,
                saveat=diffrax.SaveAt(t1=True, fn=self._max_save_fn),
                throw=False,
                max_steps=self.max_steps,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),  # exactly two items
            )

            sk_aug = jnp.atleast_2d(solk.ys)   # (n_saved, 2n+n^2), usually (1, ...)
            last   = sk_aug[-1]                # (2n+n^2,)

            yk   = last[:n]
            yMax = last[n:2*n]
            Gk   = last[2*n:].reshape(n, n)
            return yk, yMax, Gk

        def _continue(carry):
            k, done, *_ = carry
            return (~done) & (k < self.max_shooting_iterations)

        def _one_iter(carry):
            k, done, s, _, _ = carry

            t0s, t1s = ts[:-1], ts[1:]
            yk, yMax, Gk = jax.vmap(_integrate_segment, in_axes=(0, 0, 0))(s, t0s, t1s)

            # Residuals for current iterate s
            F_cont = yk[:-1] - s[1:]   # (m-1, n)
            F_last = yk[-1]  - s[0]    # (n,)

            # Build condensed linear system for Newton step Δs
            def _prod_step(P, Gi):
                return Gi @ P, None
            P, _ = jax.lax.scan(_prod_step, eye_N, Gk[:-1])

            G_next = Gk[1:][::-1]
            F_rev  = F_cont[::-1]
            def _suffix_step(acc, gf):
                Gnx, Fj = gf
                return Gnx @ acc + Fj, None
            S, _ = jax.lax.scan(_suffix_step, jnp.zeros((n,), dtype=s.dtype), (G_next, F_rev))

            Jc  = (-eye_N) + (Gk[-1] @ P)
            rhs = -(F_last + Gk[-1] @ S)
            delta_s0 = jnp.linalg.solve(Jc, rhs)

            def _prop_step(ds_prev, gf):
                Gi, Fi = gf
                ds_next = Gi @ ds_prev + Fi
                return ds_next, ds_next
            _, ds_tail = jax.lax.scan(_prop_step, delta_s0, (Gk[:-1], F_cont))
            delta_s = jnp.vstack((delta_s0, ds_tail))   # (m, n)

            s_new = s + delta_s

            # ---------- Convergence check (∞-norm), NEW ----------
            resid_inf = jnp.maximum(jnp.max(jnp.abs(F_cont)), jnp.max(jnp.abs(F_last)))
            step_inf  = jnp.max(jnp.abs(delta_s))

            new_done = (resid_inf <= self.feas_tol) & (step_inf <= self.step_tol)

            # If converged, keep s (don’t apply the update); otherwise accept s_new
            s_out = jax.lax.select(new_done, s, s_new)
            # -----------------------------------------------------

            return (k + 1, new_done, s_out, yMax, Gk)

        init = (
            jnp.array(0, jnp.int32),
            jnp.array(False),
            s,
            jnp.zeros_like(s),                   # yMax
            jnp.tile(eye_N[None, ...], (m, 1, 1))# Gk
        )
        k, converged, s_final, yMax, Gk = jax.lax.while_loop(_continue, _one_iter, init)

        y_max_all = yMax
        max_displacement = jnp.max(jnp.abs(y_max_all[:, :self.model.n_modes]), axis=0)
        max_displacement = jnp.where(converged, max_displacement, jnp.nan * max_displacement)

        y0 = s_final[0]
        y0 = jnp.where(converged, y0, jnp.nan * y0)

        # Monodromy M = G_{m-1} ... G_0
        def _prod_all(P, Gi):
            return Gi @ P, None
        M, _ = jax.lax.scan(_prod_all, jnp.eye(n, dtype=y0.dtype), Gk)
        mu = jnp.linalg.eigvals(M)

        return y0, max_displacement, mu
