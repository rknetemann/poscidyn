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
from .utils.polynomials import LagrangeBasis

class CollocationSolver(AbstractSolver):
    def __init__(self,  max_iterations: int = 20, N_elements: int = 10, K_polynomial_degree: int = 2, multistart: AbstractMultistart = LinearResponseMultistart(),
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, verbose: bool = False):

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

        self.model: AbstractModel = None

        self.multistart.verbose = self.verbose

        self.T = 2.0 * jnp.pi
        self.t0 = 0.0
        self.t1 = self.T
        
        self.ts = jnp.linspace(self.t0, self.t1, self.N_elements * (self.K_polynomial_degree + 1))  # (N_elements*(K_polynomial_degree+1),)
        self.ts_segments = self.ts.reshape(self.N_elements, self.K_polynomial_degree + 1)  # (N_elements, K_polynomial_degree+1)
        self.he = (self.t1 - self.t0) / self.N_elements 
        
        self.tau = jnp.linspace(0, 1, self.K_polynomial_degree + 1)  # (K_polynomial_degree+1,)
        self.lagrange_basis = LagrangeBasis(self.tau)
        self.f = jax.vmap(self._rhs, in_axes=(0, 0, None))  # (K+1, n_modes*2)
   
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
    
    def _calc_periodic_solution(self,
            driving_frequency: jax.Array,
            driving_amplitude: jax.Array,
            y0_guess: jax.Array):
        
        args = driving_amplitude, driving_frequency
                
        Y0 = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs), solver=diffrax.Kvaerno5(),
            t0=self.t0, t1=self.t1, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=self.ts),
            throw=False,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_amplitude, driving_frequency),
        ).ys  # (N_elements*(K_polynomial_degree+1), n_modes * 2)
                
        solver = optx.LevenbergMarquardt(rtol=self.rtol, atol=self.atol, norm=optx.rms_norm, verbose=frozenset({"step", "accepted", "loss", "step_size"})) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
        sol = optx.least_squares(self._residual, solver, args=args, y0=Y0, options={"jac": "bwd"}, max_steps=self.max_iterations, throw=True) 
        
        ### TEMPORARY PLOTTING CODE ###
        Y = sol.value
        x = Y[0, :]  # (n_modes*2,)
        Y_series = Y.reshape(-1, self.model.n_modes * 2)
        ts = self.ts
        x_series = Y_series[:, 0]

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(ts, x_series)
        # plt.xlabel('Time')
        # plt.ylabel('Displacement x')
        # plt.title('Periodic Solution – x vs. t')
        # plt.show()
        y_max = jnp.max(jnp.abs(Y_series[:, 0::2]), axis=0)
        return Y[0], y_max, None

    def _residual(self, Y, args):
        Y_segments = Y.reshape(self.N_elements, self.K_polynomial_degree + 1, -1)  # (N_elements, K+1, n_modes*2)
        
        def _collocation_residual(ts_segments, Y_segments):
            def _one_segment(ts_i, Y_i):
                dz_dtau = self.lagrange_basis.differentiation_matrix() @ Y_i  # (K+1, n_modes*2)
                dy_dt = (1.0 / self.he) * dz_dtau   # (K+1, n_modes*2)

                dy_dt_eff = dy_dt[1:, :]            # rows 1..K
                t_i_eff   = ts_i[1:]                # same rows
                r_col = dy_dt_eff - self.f(t_i_eff, Y_i[1:, :], args)  # (K, n)

                return r_col
            
            _batch_segments = jax.vmap(_one_segment, in_axes=(0, 0))
            R_col = _batch_segments(ts_segments, Y_segments)  # (N_elements, K+1, n_modes*2)
            return R_col
        
        def _continuity_residual(Y_segments):
            R_cont = Y_segments[1:, 0, :] - Y_segments[:-1, -1, :]  # (N_elements-1, n_modes*2)
            return R_cont
        
        def _periodicity_residual(Y_segments):
            R_per = Y_segments[0, 0, :] - Y_segments[-1, -1, :]  # (n_modes*2,)
            return R_per

        R_col = _collocation_residual(self.ts_segments, Y_segments)  # (N_elements, K+1, n_modes*2)
        R_cont = _continuity_residual(Y_segments)               # (N_elements-1, n_modes*2)
        R_per = _periodicity_residual(Y_segments)               # (n_modes*2,)

        R = jnp.concatenate([R_col.flatten(), R_cont.flatten(), R_per.flatten()])  # (n_modes*2*N_elements*(K+1),)
        return R
    
    def _rhs(self, t, y, args):
        # args = (drive_amp, drive_freq); pass them through to the model
        return self.model.f(t, y, args)
