import jax
import jax.numpy as jnp
import diffrax
from functools import partial
from equinox import filter_jit

from .abstract_solver import AbstractSolver
from ..models.abstract_model import AbstractModel
from ..utils.plotting import plot_branch_exploration, plot_branch_selection
from .utils.coarse_grid import gen_coarse_grid_1, gen_coarse_grid_2
from .utils.branch_selection import select_branches
from .utils.uique_solutions import get_unique_solutions
from .. import constants as const 

# Optional plotting import
try:
    import numpy as _np
    import matplotlib.pyplot as _plt
except Exception:
    _plt = None
    _np = None

class MultipleShootingSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096,
                 max_shooting_iterations: int = 20, shooting_tolerance: float = 1e-10,
                 m_segments: int = 20, max_branches: int = 5,
                 rtol: float = 1e-4, atol: float = 1e-7, progress_bar: bool = False):
        super().__init__(rtol=rtol, atol=atol, max_steps=max_steps)

        self.n_time_steps = n_time_steps
        self.max_shooting_iterations = max_shooting_iterations
        self.shooting_tolerance = shooting_tolerance
        self.m_segments = m_segments
        self.max_branches = max_branches
        self.progress_bar = progress_bar
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
                 init_guess: jax.Array,  
                ):
        
        self.n = self.model.n_modes * 2
        self.m = self.m_segments
                       
        y0 = self._calculate_period_solution(drive_freq, drive_amp, init_guess)

        def _solve_one_period(y0):               
            sol = self._solve(self._rhs, True, self.t0, self.t1, y0, drive_freq, drive_amp)
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

        # Build coarse grid
        coarse_drive_freq_mesh, coarse_drive_amp_mesh, coarse_init_disp_mesh, coarse_init_vel_mesh = gen_coarse_grid_2(
            self.model, drive_freq, drive_amp
        )

        # Flatten for vmap; keep first-dimension = number of simulations.
        coarse_drive_freq_flat = coarse_drive_freq_mesh.ravel()                       # (n_sim,)
        coarse_drive_amp_flat  = coarse_drive_amp_mesh.reshape(coarse_drive_amp_mesh.shape[0], -1).ravel() \
                                if coarse_drive_amp_mesh.ndim == 2 else coarse_drive_amp_mesh.ravel()
        # ^ If your amp mesh already has shape (n_sim,), this is a no-op.
        coarse_init_disp_flat  = coarse_init_disp_mesh.ravel()                        # (n_sim,)
        coarse_init_vel_flat   = coarse_init_vel_mesh.ravel()                         # (n_sim,)

        # Solve shooting for each coarse combination to get y0
        @jax.jit
        def shooting_case(drive_freq, drive_amp, init_disp, init_vel):
            init_disp = jnp.full((self.model.n_modes,), init_disp)
            init_vel  = jnp.full((self.model.n_modes,),  init_vel)
            init_cond = jnp.concatenate([init_disp, init_vel])
            return self._calculate_period_solution(drive_freq, drive_amp, init_cond)

        y0, y_max = jax.vmap(shooting_case)(
            coarse_drive_freq_flat, coarse_drive_amp_flat, coarse_init_disp_flat, coarse_init_vel_flat
        )  # y0: (n_sim, n)  y_max: (n_sim, n_modes)

        print(y_max.shape)  # (n_coarse_drive_freq * n_coarse_drive_amp * n_ic, n_modes)

        # PLOT HERE
        if _plt is not None and _np is not None:
            freqs_np = _np.asarray(coarse_drive_freq_flat)
            amps_np  = _np.asarray(coarse_drive_amp_flat)
            y_max_np = _np.asarray(y_max)  # (n_sim, n_modes)

            unique_amps = _np.unique(amps_np)
            n_modes = y_max_np.shape[1]

            fig, axes = (_plt.subplots(n_modes, 1, sharex=True, figsize=(6, 3 * n_modes))
                         if n_modes > 1 else (None, [_plt.subplots(1, 1, figsize=(6, 4))[1]]))

            for mode in range(n_modes):
                ax = axes[mode] if n_modes > 1 else axes[0]
                for amp in unique_amps:
                    mask = amps_np == amp
                    freqs_masked = freqs_np[mask]
                    y_mode = y_max_np[mask, mode]

                    # Aggregate duplicates: max over same frequency
                    uf = _np.unique(freqs_masked)
                    y_agg = _np.array([y_mode[freqs_masked == f].max() for f in uf])

                    ax.plot(uf, y_agg, marker='o', linestyle='-', label=f"A={amp:.3g}")
                ax.set_ylabel(f"y_max (mode {mode+1})")
                ax.grid(alpha=0.3)

            axes[-1].set_xlabel("Drive frequency")
            axes[0].set_title("Maximum periodic displacement vs drive frequency")
            axes[0].legend(ncol=2, fontsize='small')
            _plt.tight_layout()
            _plt.show()
        else:
            print("matplotlib not available; skipping plot.")

        @jax.jit
        def _solve_one_period(y0_single, freq_single, amp_single):
            def do_integrate(y_init):
                sol = self._solve(self._rhs, True, self.t0, self.t1, y_init, freq_single, amp_single)
                return sol.ts, sol.ys
            return jax.lax.cond(
                jnp.isnan(y0_single).any(),
                lambda: (jnp.zeros((self.n_time_steps,)),
                         jnp.zeros((self.n_time_steps, y0_single.size), dtype=y0_single.dtype)),
                lambda: do_integrate(y0_single)
            )

        ts, ys = jax.vmap(_solve_one_period)(y0, coarse_drive_freq_flat, coarse_drive_amp_flat) # (n_sim, n_time_steps), (n_sim, n_time_steps, n_modes*2)


        

        return drive_freq, y_max_disp

    
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

    def _calculate_period_solution(self,
                 driving_frequency: jax.Array, 
                 driving_amplitude: jax.Array,  
                 initial_condition: jax.Array, 
                ):     

        T = 2.0 * jnp.pi
        ts = jnp.linspace(0.0, T, self.m + 1)
        s0 = jnp.tile(initial_condition, (self.m, 1))  # (m, n)

        eye_N = jnp.eye(self.n, dtype=initial_condition.dtype)
            
        @filter_jit
        def _integrate_segment(sk, t0k, t1k):
            X0 = eye_N.reshape(-1)
            y_max0 = sk
            sk0_aug = jnp.hstack([sk, y_max0, X0], dtype=initial_condition.dtype)  # Augmented state: [y; vec(X)]
            sk_aug = self._solve(self._aug_rhs, False, t0k, t1k, sk0_aug, driving_frequency, driving_amplitude).ys

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

