import jax
import jax.numpy as jnp
import diffrax
from functools import partial
from equinox import filter_jit

from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from ..utils.plotting import plot_branch_exploration, plot_branch_selection
from .utils.coarse_grid import gen_coarse_grid_1
from .utils.branch_selection import select_branches
from .. import constants as const 

class SingleShootingSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096,
                 max_shooting_iterations: int = 20,
                 shooting_tolerance: float = 1e-10,
                 rtol: float = 1e-4, atol: float = 1e-7, progress_bar: bool = False):

        self.n_time_steps = n_time_steps
        self.max_shooting_iterations = max_shooting_iterations
        self.shooting_tolerance = shooting_tolerance
        self.progress_bar = progress_bar
        self.model: AbstractModel = None
   
    def time_response(self,
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_guess: jax.Array,  
                ):
                       
        y0 = self._calculate_period_solution(drive_freq, drive_amp, init_guess)

        def _solve_one_period(y0):               
            sol = self._solve(self._rhs, True, y0, drive_freq, drive_amp)
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
                 drive_amp: jax.Array,   # (n_modes,)
                 sweep_direction: const.SweepDirection,
                ):
    
        # Generate combinations of coarse driving frequencies, amplitudes, and initial conditions
        # where the intiial conditions are based on the linear response amplitude for the given driving frequency and amplitude ranges
        coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh, coarse_init_vel_mesh = gen_coarse_grid(
            self.model, drive_freq, drive_amp
        )

        # Flatten everything to 1D arrays for vmap
        coarse_drive_freq_flat = coarse_drive_freq_mesh.ravel()
        coarse_drive_amp_flat  = coarse_drive_amp_mesh.ravel()
        coarse_init_disp_flat  = coarse_init_disp_mesh.ravel()
        coarse_init_vel_flat   = coarse_init_vel_mesh.ravel()
        
        # Define a jitted function to solve each combination of parameters
        @jax.jit
        def shooting_case(drive_freq, drive_amp, init_disp, init_vel):
            init_disp = jnp.full((self.model.n_modes,), init_disp)
            init_vel = jnp.full((self.model.n_modes,), init_vel)
            init_cond = jnp.concatenate([init_disp, init_vel])

            return self._calculate_period_solution(drive_freq, drive_amp, init_cond)

        y0 = jax.vmap(shooting_case)(coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat)

        @jax.jit
        def _solve_one_period(y0_single, freq_single, amp_single):
            def do_integrate(y_init):
                sol = self._solve(self._rhs, True, y_init, freq_single, amp_single)
                return sol.ts, sol.ys
            return jax.lax.cond(
                jnp.isnan(y0_single).any(),
                lambda: (jnp.zeros((self.n_time_steps,)), jnp.zeros((self.n_time_steps, y0_single.size), dtype=y0_single.dtype)),
                lambda: do_integrate(y0_single)
            )

        ts, ys = jax.vmap(_solve_one_period)(y0, coarse_drive_freq_flat, coarse_drive_amp_flat) # (n_sim, n_time_steps), (n_sim, n_time_steps, n_modes*2)

        ss_time_flat = ts # (n_sim, n_time_steps)
        ss_disp_flat = ys[..., :self.model.n_modes] # (n_sim, n_time_steps, n_modes)
        ss_vel_flat = ys[..., self.model.n_modes:] # (n_sim, n_time_steps, n_modes)

        idx_max_disp = jnp.argmax(jnp.abs(ss_disp_flat), axis=1) # (n_sim, n_modes)
        t_max_disp = jnp.take_along_axis(ss_time_flat[:, :, None], idx_max_disp[:, None, :], axis=1).squeeze(1) # (n_sim, n_modes)
        q_max_disp = jnp.abs(jnp.take_along_axis(ss_disp_flat, idx_max_disp[:, None, :], axis=1).squeeze(1)) # (n_sim, n_modes)
        v_max_disp = jnp.abs(jnp.take_along_axis(ss_vel_flat, idx_max_disp[:, None, :], axis=1).squeeze(1)) # (n_sim, n_modes)
        y_max_disp = jnp.concatenate([q_max_disp, v_max_disp], axis=-1) # (n_sim, n_modes * 2)

        idx_max_vel = jnp.argmax(jnp.abs(ss_vel_flat), axis=1) # (n_sim, n_modes)
        t_max_vel = jnp.take_along_axis(ss_time_flat[:, :, None], idx_max_vel[:, None, :], axis=1).squeeze(1) # (n_sim, n_modes)
        q_max_vel = jnp.abs(jnp.take_along_axis(ss_disp_flat, idx_max_vel[:, None, :], axis=1).squeeze(1))  # (n_sim, n_modes)
        v_max_vel = jnp.abs(jnp.take_along_axis(ss_vel_flat, idx_max_vel[:, None, :], axis=1).squeeze(1))  # (n_sim, n_modes)
        y_max_vel = jnp.concatenate([q_max_vel, v_max_vel], axis=-1) # (n_sim, n_modes * 2)

        t_max_disp = t_max_disp.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            self.model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)

        y_max_disp = y_max_disp.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            self.model.n_modes * 2
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)

        t_max_vel = t_max_vel.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            self.model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)  

        y_max_vel = y_max_vel.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            self.model.n_modes * 2
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)
        
        plot_branch_exploration(coarse_drive_freq_mesh, coarse_drive_amp_mesh, y_max_disp)

        t_max_disp_sel, y_max_disp_sel = select_branches(t_max_disp, y_max_disp, sweep_direction.value)
        ss_disp_amp = jnp.abs(y_max_disp_sel[..., :self.model.n_modes])   # (n_coarse_freq, n_coarse_amp, n_modes)

        plot_branch_selection(drive_freq, drive_amp, ss_disp_amp)

        return drive_freq, y_max_disp
    
    @filter_jit
    def _rhs(self, tau, y, args):
        _drive_amp, drive_freq  = args

        t = tau / drive_freq
        dydt = self.model.f(t, y, args) / drive_freq

        return dydt
    
    @filter_jit
    def _aug_rhs(self, tau, y_aug, args):
        _drive_amp, drive_freq  = args

        y  = y_aug[:2]
        X  = y_aug[2:].reshape(2, 2)

        t = tau/ drive_freq
        f = self.model.f(t, y, args)      
        f_y  = self.model.f_y(t, y, args)

        dydt = f / drive_freq
        dXdt = f_y @ X / drive_freq
        return jnp.hstack([dydt, dXdt.reshape(-1)])

    def _calculate_period_solution(self,
                 driving_frequency: jax.Array, 
                 driving_amplitude: jax.Array,  
                 initial_condition: jax.Array, 
                ):        

        @filter_jit
        def _newton_shooting(y0):
            X0 = jnp.eye(2, dtype=initial_condition.dtype).reshape(-1)
            y0_aug = jnp.hstack([y0, X0], dtype=initial_condition.dtype)  # Augmented state: [y; vec(X)]

            ys_aug = self._solve(self._aug_rhs, False, y0_aug, driving_frequency, driving_amplitude).ys
            ys = ys_aug[:, :2]
            yT = ys_aug[-1, :2]
            XT = ys_aug[-1, 2:].reshape(2, 2)
            
            F = yT - y0
            J = XT - jnp.eye(2, dtype=initial_condition.dtype)
            
            r = jnp.linalg.norm(F, ord=jnp.inf)

            return y0, yT, F, J, XT, r
        
        def newton_step(J, F):
            # Condition proxy: determinant for 2x2 or cond if you prefer
            det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            near_sing = jnp.abs(det) < 1e-10  # tune as needed

            def reg():
                lam = 1e-8
                JTJ = J.T @ J + lam * jnp.eye(2, dtype=J.dtype)
                return jnp.linalg.solve(JTJ, -J.T @ F)

            def direct():
                return jnp.linalg.solve(J, -F)

            return jax.lax.cond(near_sing, reg, direct)

        def _shooting_converged_cond(carry):
            k, done, y0, yT, XT, r = carry
            return jnp.logical_and(~done, k < self.max_shooting_iterations)

        def _shooting_iteration(carry):
            k, done, y0, yT, XT, r = carry

            y0, yT, F, J, XT, r = _newton_shooting(y0)
                      
            dy0 = newton_step(J, F)
            
            def _line_search_cond(carry):
                i, lam, done, y0, yT, XT, r = carry
                return jnp.logical_and(~done, i < 8)

            def _line_search_iteration(carry):
                i, lam, ls_done, y0, yT, mu, r = carry

                y0_try = y0 + lam * dy0
                y0_try, yT_try, F_try, J_try, mu_try, r_try = _newton_shooting(y0_try)

                success = r_try < 0.7 * r

                # Accept step if successful; otherwise halve step size and continue
               
                lam = jnp.where(success, lam, lam * 0.5)
                ls_done = jnp.logical_or(ls_done, success)

                y0 = jnp.where(success, y0_try, y0)
                yT = jnp.where(success, yT_try, yT)
                mu = jnp.where(success, mu_try, mu)
                r = jnp.where(success, r_try, r)

                i += 1

                return (i, lam, ls_done, y0, yT, XT, r)          
            
            init_carry = (
                jnp.array(0, dtype=jnp.int32),           # i
                jnp.array(1.0, dtype=y0.dtype),          # lam
                jnp.array(False),                        # ls_done
                y0,                                      # y0
                yT,                                      # yT
                jnp.eye(2, dtype=y0.dtype),              # XT
                r,                                       # r
            )

            i, lam, ls_done, y0_after_ls, yT, XT, r = jax.lax.while_loop(
                _line_search_cond, _line_search_iteration, init_carry
            )

            # for-else fallback: if never succeeded, use last resort y0 + dy0
            y0 = jnp.where(ls_done, y0_after_ls, y0 + dy0)  

            sh_done = r < self.shooting_tolerance
            k += 1
                        
            return k, sh_done, y0, yT, XT, r
        
        init_carry = (
            jnp.array(0, dtype=jnp.int32), # k
            jnp.array(False), # done
            initial_condition, # y0: (2,)
            jnp.full((2,), jnp.inf, dtype=initial_condition.dtype), # yT: (2,)
            jnp.eye(2, dtype=initial_condition.dtype), # XT: (2,2)
            jnp.array(jnp.inf, dtype=initial_condition.dtype), # r: scalar float
        )

        k, ls_done, y0, yT, XT, r = jax.lax.while_loop(_shooting_converged_cond, _shooting_iteration, init_carry)

        y0 = jnp.where(ls_done, y0, jnp.nan) 
        #mu = jnp.linalg.eigvals(XT)
        #jax.debug.print("Converged: {}", ls_done)

        return y0  # Return time and state (displacement and velocity)

    def _solve(self,
               rhs: callable,
               wf: bool,
               y0: jax.Array,
               driving_frequency: float,
               driving_amplitude: jax.Array) -> diffrax.Solution:
        
        dtmax = self.T / const.DT_MAX_FACTOR
        dtmin = self.T / const.DT_MIN_FACTOR
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(rhs),
            solver=diffrax.Tsit5(),
            t0=self.t0,
            t1=self.t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, self.T, self.n_time_steps)) if wf else diffrax.SaveAt(t1=True),
            throw=False,
            progress_meter=diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, dtmax=dtmax, dtmin=dtmin
            ),
            args=(driving_amplitude, driving_frequency),
        )
        return sol
    
