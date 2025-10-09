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
        
        pass