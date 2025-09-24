import jax
import jax.numpy as jnp
import diffrax

from .abstract_solver import AbstractSolver
from ..models import AbstractModel
from ..utils.plotting import plot_branch_exploration
from .utils.coarse_grid import gen_coarse_grid
from .. import constants as const 

class ShootingSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096,
                 rtol: float = 1e-6, atol: float = 1e-6, progress_bar: bool = False):
        super().__init__(rtol=rtol, atol=atol, max_steps=max_steps)
        self.n_time_steps = n_time_steps
        self.progress_bar = progress_bar
    
    def time_response(self,
                 model: AbstractModel,
                 drive_freq: float,   # shape (1,) or scalar
                 drive_amp: float,   # shape (n_modes,) — here we assume 1 mode
                 init_guess: jax.Array,   # shape (2,) for single-mode Duffing)
                ):
                
        ts, ys = self._calculate_period_solution(model, drive_freq, drive_amp, init_guess)
        return ts, ys
    
    def frequency_sweep(self,
                 model: AbstractModel,
                 drive_freq: jax.Array,   # (1,) or scalar
                 drive_amp: jax.Array,   # (n_modes,)
                ):
    
        coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat = gen_coarse_grid(
            model, drive_freq, drive_amp
        )
        
        @jax.jit
        def solve_case(drive_freq, drive_amp, init_disp, init_vel):
            init_disp = jnp.full((model.n_modes,), init_disp)
            init_vel = jnp.full((model.n_modes,), init_vel)
            init_cond = jnp.concatenate([init_disp, init_vel])

            return self._calculate_period_solution(model, drive_freq, drive_amp, init_cond)

        sol = jax.vmap(solve_case)(coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat)
        
        ts, ys = sol

        ss_time_flat = ts # (n_sim, n_time_steps)
        ss_disp_flat = ys[..., :model.n_modes] # (n_sim, n_time_steps, n_modes)
        ss_vel_flat = ys[..., model.n_modes:] # (n_sim, n_time_steps, n_modes)

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
            model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)

        y_max_disp = y_max_disp.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            model.n_modes * 2
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)

        t_max_vel = t_max_vel.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            model.n_modes
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)  

        y_max_vel = y_max_vel.reshape(
            const.N_COARSE_DRIVING_FREQUENCIES, 
            const.N_COARSE_DRIVING_AMPLITUDES, 
            const.N_COARSE_INITIAL_DISPLACEMENTS, 
            const.N_COARSE_INITIAL_VELOCITIES,
            model.n_modes * 2
        ) # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes * 2)
        
        plot_branch_exploration(coarse_drive_freq_mesh, coarse_drive_amp_mesh, y_max_disp)

        return drive_freq, y_max_disp


    def _calculate_period_solution(self,
                 model: AbstractModel,
                 driving_frequency: jax.Array,   # shape (1,) or scalar
                 driving_amplitude: jax.Array,   # shape (n_modes,) — here we assume 1 mode
                 initial_condition: jax.Array,   # shape (2,) for single-mode Duffing
                ):
        
        T = const.MAXIMUM_ORDER_SUBHARMONICS * 2.0 * jnp.pi / driving_frequency  # Period of oscillation including subharmonics
        t0 = 0.0
        t1 = T
        ts = jnp.linspace(0.0, T, self.n_time_steps)
                
        TOL = 1e-4

        @jax.jit
        def _newton_shooting(y0):
            X0 = jnp.eye(2, dtype=initial_condition.dtype).reshape(-1)
            y0_aug = jnp.hstack([y0, X0])

            ys_aug = self._solve(model, t0, t1, ts, y0_aug, driving_frequency, driving_amplitude).ys
            ys = ys_aug[:, :2]
            yT = ys_aug[-1, :2]
            XT = ys_aug[-1, 2:].reshape(2, 2)
            
            F = yT - y0
            J = XT - jnp.eye(2)
            
            r = jnp.linalg.norm(F, ord=jnp.inf)
            
            mu = jnp.linalg.eigvals(XT)

            return y0, yT, ys, F, J, mu, r

        @jax.jit
        def _shooting_converged_cond(carry):
            k, done, y0, yT, ys, mu, r = carry
            return jnp.logical_and(~done, k < const.MAX_SHOOTING_ITERATIONS)

        @jax.jit
        def _shooting_iteration(carry):
            k, done, y0, yT, ys, mu, r = carry

            y0, yT, ys, F, J, mu, r = _newton_shooting(y0)
                      
            dy0 = jnp.linalg.solve(J, -F)
            
            def _line_search_cond(carry):
                i, lam, done, y0, yT, ys, mu, r = carry
                return jnp.logical_and(~done, i < 8)

            def _line_search_iteration(carry):
                i, lam, ls_done, y0, yT, ys, mu, r = carry

                y0_try = y0 + lam * dy0
                y0_try, yT_try, ys_try, F_try, J_try, mu_try, r_try = _newton_shooting(y0_try)

                success = r_try < 0.7 * r

                # Accept step if successful; otherwise halve step size and continue
               
                lam = jnp.where(success, lam, lam * 0.5)
                ls_done = jnp.logical_or(ls_done, success)

                y0 = jnp.where(success, y0_try, y0)
                yT = jnp.where(success, yT_try, yT)
                ys = jnp.where(success, ys_try, ys)
                mu = jnp.where(success, mu_try, mu)
                r = jnp.where(success, r_try, r)

                i += 1

                return (i, lam, ls_done, y0, yT, ys, mu, r)          
            
            init_carry = (
                jnp.array(0, dtype=jnp.int32),           # i
                jnp.array(1.0, dtype=y0.dtype),          # lam
                jnp.array(False),                        # ls_done
                y0,                                      # y0
                yT,                                      # yT
                ys,                                      # ys
                mu,                                      # mu
                r,                                       # r
            )

            i, lam, ls_done, y0_after_ls, yT, ys, mu, r = jax.lax.while_loop(
                _line_search_cond, _line_search_iteration, init_carry
            )

            # for-else fallback: if never succeeded, use last resort y0 + dy0
            y0 = jnp.where(ls_done, y0_after_ls, y0 + dy0)  

            sh_done = r < TOL
            k += 1
                        
            return k, sh_done, y0, yT, ys, mu, r
        
        init_carry = (
            jnp.array(0, dtype=jnp.int32), # k
            jnp.array(False), # done
            initial_condition, # y0: (2,)
            jnp.full((2,), jnp.inf, dtype=initial_condition.dtype), # yT: (2,)
            jnp.zeros((self.n_time_steps, 2), dtype=initial_condition.dtype), # ys: (N,6)
            jnp.zeros((2,), dtype=jnp.result_type(initial_condition.dtype, jnp.complex64)), # mu: (2,), complex
            jnp.array(jnp.inf, dtype=initial_condition.dtype), # r: scalar float
        )

        k, ls_done, y0, yT, ys, mu, r = jax.lax.while_loop(_shooting_converged_cond, _shooting_iteration, init_carry)

        ys = jnp.where(ls_done, ys, jnp.nan) 
        #jax.debug.print("Converged: {}", ls_done)

        return ts, ys  # Return time and state (displacement and velocity)

    def _solve(self,
               model: AbstractModel,
               t0: float,
               t1: float,
               ts: jax.Array,
               y0: jax.Array,
               driving_frequency: float,
               driving_amplitude: jax.Array) -> diffrax.Solution:

        @jax.jit
        def aug_rhs(t, y_aug, args):
            y  = y_aug[:2]
            X  = y_aug[2:].reshape(2, 2)

            f = model.f(t, y, args)      
            f_y  = model.f_y(t, y, args)

            dydt = f
            dXdt = f_y @ X
            return jnp.hstack([dydt, dXdt.reshape(-1)])

        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(aug_rhs),
            solver=diffrax.Tsit5(),
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
            t0=t0,
            t1=t1,
            dt0=None,
            max_steps=self.max_steps,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
            progress_meter=diffrax.TqdmProgressMeter() if self.progress_bar else diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol, atol=self.atol, pcoeff=0.0, icoeff=1.0, dcoeff=0.0
            ),
            args=(driving_amplitude, driving_frequency),
        )
        return sol