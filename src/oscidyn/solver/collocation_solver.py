import jax
import jax.numpy as jnp
import diffrax
from equinox import filter_jit
import optimistix as optx
import lineax as lx

from .abstract_solver import AbstractSolver
from ..model.abstract_model import AbstractModel
from .multistart.abstract_multistart import AbstractMultistart
from .multistart.linear_response_multistart import LinearResponseMultistart
from .. import constants as const 

class CollocationSolver(AbstractSolver):
    def __init__(self,  max_iterations: int = 20, n_collocation_points: int = 10, multistart: AbstractMultistart = LinearResponseMultistart(),
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, verbose: bool = False):

        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.n_collocation_points = n_collocation_points
        self.order_approx = 5
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol
        self.verbose = verbose

        self.model: AbstractModel = None

        self.multistart.verbose = self.verbose

        self.T = 2.0 * jnp.pi
        self.t0 = 0.0
        self.t1 = self.T * 3.0  # 3 forcing periods for 1/3 subharmonic
   
    def time_response(self,
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_disp: jax.Array,  
                 init_vel: jax.Array
                ):

        y0_guess = jnp.array([init_disp, init_vel]).flatten()

        y0, _, _ = self._calc_periodic_solution(drive_freq, drive_amp, y0_guess)

        def _solve_one_period(y0):               
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Tsit5(),
                t0=self.t0, t1=self.t1, dt0=None, max_steps=self.max_steps,
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(self.t0, self.t1, self.n_time_steps)),
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
             drive_freq: jax.Array,
             drive_amp: jax.Array, 
             sweep_direction: const.SweepDirection,
            ):

        drive_freq_mesh, drive_amp_mesh, init_disp_mesh, init_vel_mesh = self.multistart.generate_simulation_grid(
            self.model, drive_freq, drive_amp
        )
        drive_freq_mesh_flat = drive_freq_mesh.ravel()                       # (n_sim,)
        drive_amp_mesh_flat  = drive_amp_mesh.ravel()                        # (n_sim,)

        y0_guess = jnp.stack([init_disp_mesh, init_vel_mesh], axis=-1)  # (F, A, D, V, 2)
        y0_guess = y0_guess.reshape(-1, 2)                              # (n_sim, 2) for 1 mode

        periodic_solutions = jax.vmap(self._calc_periodic_solution)(
            drive_freq_mesh_flat, drive_amp_mesh_flat, y0_guess
        )  # y0: (n_sim, n_modes*2)  y_max: (n_sim, n_modes)

        return periodic_solutions

    @filter_jit
    def _calc_periodic_solution(self,
            driving_frequency: jax.Array,
            driving_amplitude: jax.Array,
            y0_guess: jax.Array):

        ts = jnp.linspace(self.t0, self.t1, self.n_collocation_points)
        args = (driving_amplitude, driving_frequency)

        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs),
            solver=diffrax.Tsit5(),
            t0=self.t0, t1=self.t1, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=ts),
            throw=False,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=args,
        )

        x_samp = sol.ys[:, :self.model.n_modes].squeeze()  # (n_collocation_points,)
        
        def _residual(coeff):
            # Approximate y and y' wrt tau from the ansatz
            x_t = self.x(ts, coeff, args)                                  # (M,)
            x_t_dot = self.dx_dtau(ts, coeff, args)                        # (M,)
            x_t_ddot = self.d2x_dtau2(ts, coeff, args)                     # (M,)

            # Convert to physical variables in state y=[x, v] where v = dx/dt = drive_freq * dx/dtau
            v_t = driving_frequency * x_t_dot                              # (M,)
            y_t = jnp.stack([x_t, v_t], axis=1)                          # (M, 2)

            # y'_approx wrt tau: [dx/dtau, d/dtau v] with v = w * dx/dtau => dv/dtau = w * d2x/dtau2
            y_t_dot = jnp.stack([x_t_dot, driving_frequency * x_t_ddot], axis=1)  # (M, 2)

            # ODE RHS already scaled to tau: y' = self._rhs(tau, y, args)
            rhs = jax.vmap(lambda tt, yy: self._rhs(tt, yy, args))(ts, y_t)         # (M, 2)

            r = (y_t_dot - rhs).reshape(-1)                                   # (2M,)
            return r
        
        solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8, norm=optx.rms_norm) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
        sol = optx.least_squares(_residual, solver, y0=s0, options={"jac": "bwd"}, max_steps=self.max_shooting_iterations, throw=False) 


       
        # ---- Step 4: build outputs from final coeff ----
        # y0 at tau=0
        tau_zero = jnp.array([0.0], dtype=taus.dtype)
        x0 = self.x(tau_zero, coeff, args)[0]
        x0_dtau = self.dx_dtau(tau_zero, coeff, args)[0]
        v0 = driving_frequency * x0_dtau
        y0 = jnp.array([x0, v0])

        # estimate peak |x| over three forcing periods from a denser grid
        tau_fine = jnp.linspace(tau0, tau1, 5 * M)
        x_fine = self.x(tau_fine, coeff, args)
        y_max = jnp.array([jnp.max(jnp.abs(x_fine))])

        # If residual is too large or NaN, return NaNs so callers can skip
        r_final = residual(coeff)
        bad = jnp.logical_or(jnp.isnan(r_final).any(), jnp.linalg.norm(r_final) > 1e-3)
        y0 = jax.lax.cond(bad, lambda: jnp.array([jnp.nan, jnp.nan]), lambda: y0)
        y_max = jax.lax.cond(bad, lambda: jnp.array([jnp.nan]), lambda: y_max)

        return y0, y_max, coeff       


    @filter_jit
    def x(self, tau: jax.Array, coeff: jax.Array, args):
        """1/3 subharmonic truncated odd Fourier series for displacement x(t).
        tau: phase-time in [0, 2π] for one forcing period; for 1/3 subharmonic we’ll use up to 6π.
        coeff: [a_1..a_N, b_1..b_N]
        """
        _drive_amp, drive_freq = args

        a = coeff[: self.order_approx]                # (N,)
        b = coeff[self.order_approx :]                # (N,)

        # phases = (2n-1) * tau / 3  (independent of drive_freq in tau-variable)
        n = jnp.arange(1, self.order_approx + 1)      # (N,)
        phi = ((2 * n - 1)[:, None] * (tau[None, :] / 3.0))  # (N, T)

        xc = (a[:, None] * jnp.cos(phi)).sum(axis=0)  # (T,)
        xs = (b[:, None] * jnp.sin(phi)).sum(axis=0)  # (T,)

        return xc + xs                                # (T,)


    @filter_jit
    def dx_dtau(self, tau: jax.Array, coeff: jax.Array, args):
        """Derivative of x wrt tau (NOT physical time)."""
        _drive_amp, drive_freq = args

        a = coeff[: self.order_approx]
        b = coeff[self.order_approx :]

        n = jnp.arange(1, self.order_approx + 1)
        k = (2 * n - 1) / 3.0                         # (N,)
        phi = (k[:, None] * tau[None, :])             # (N, T)

        # d/dtau [a cos(phi) + b sin(phi)] = -a k sin(phi) + b k cos(phi)
        term = (-a[:, None] * jnp.sin(phi) + b[:, None] * jnp.cos(phi)) * k[:, None]
        return term.sum(axis=0)                       # (T,)


    @filter_jit
    def d2x_dtau2(self, tau: jax.Array, coeff: jax.Array, args):
        """Second derivative of x wrt tau."""
        _drive_amp, drive_freq = args

        a = coeff[: self.order_approx]
        b = coeff[self.order_approx :]

        n = jnp.arange(1, self.order_approx + 1)
        k = (2 * n - 1) / 3.0
        k2 = k * k                                    # (N,)
        phi = (k[:, None] * tau[None, :])             # (N, T)

        # d2/dtau2 [a cos + b sin] = -(a cos + b sin) * k^2
        term = (a[:, None] * jnp.cos(phi) + b[:, None] * jnp.sin(phi)) * (-k2[:, None])
        return term.sum(axis=0)                       # (T,)

    @filter_jit
    def _rhs(self, tau, y, args):
        _drive_amp, drive_freq  = args

        t = tau / drive_freq
        dydt = self.model.f(t, y, args) / drive_freq

        return dydt
