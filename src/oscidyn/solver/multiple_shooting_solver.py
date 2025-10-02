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
from .utils.coarse_grid import gen_coarse_grid_1, gen_grid_2
from .. import constants as const 

class MultipleShootingSolver(AbstractSolver):
    def __init__(self,  max_shooting_iterations: int = 20, m_segments: int = 20, multistart: AbstractMultistart = LinearResponseMultistart(),
                 rtol: float = 1e-4, atol: float = 1e-7, n_time_steps: int = None, max_steps: int = 4096, verbose: bool = False):

        self.n_time_steps = n_time_steps
        self.max_steps = max_steps
        self.max_shooting_iterations = max_shooting_iterations
        self.m_segments = m_segments
        self.multistart = multistart
        self.rtol = rtol
        self.atol = atol

        self.model: AbstractModel = None

        self.T = 2.0 * jnp.pi
        self.t0 = 0.0
        self.t1 = self.T
   
    def time_response(self,
                 drive_freq: jax.Array,  
                 drive_amp: jax.Array, 
                 init_disp: jax.Array,  
                 init_vel: jax.Array
                ):

        y0_guess = jnp.array([init_disp, init_vel]).flatten()

        y0, x_max = self._calc_periodic_solution(drive_freq, drive_amp, y0_guess)

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

        _, x_max = jax.vmap(self._calc_periodic_solution)(
            drive_freq_mesh_flat, drive_amp_mesh_flat, y0_guess
        )  # y0: (n_sim, n_modes*2)  y_max: (n_sim, n_modes)

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

        s0 = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs), solver=diffrax.Tsit5(),
            t0=self.t0, t1=self.t1, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=jnp.linspace(self.t0, self.t1, self.m_segments + 1)),
            throw=False,
            max_steps=self.max_steps,
            progress_meter=diffrax.NoProgressMeter(),
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            args=(driving_amplitude, driving_frequency),
        ).ys[:-1]  # (m_segments, n_modes * 2)

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
            ts = jnp.linspace(self.t0, self.t1, self.m_segments + 1)
            def _one(carry, i):
                sk  = s[i]
                sk1 = s[(i + 1) % self.m_segments]    # wrap for periodicity
                Phi = _integrate_segment(sk, ts[i], ts[i + 1])  # (n_modes * 2,)
                r = Phi - sk1
                return carry, r

            _, Rs = jax.lax.scan(_one, None, jnp.arange(self.m_segments))  # Rs: (m_segments, n_modes * 2)
            return Rs.reshape((-1,))                         # (m_segments * n_modes * 2,)

        solver = optx.LevenbergMarquardt(rtol=1e-7, atol=1e-10) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
        sol = optx.least_squares(_residual, solver, y0=s0, options={"jac": "bwd"}, max_steps=self.max_shooting_iterations, throw=False) 
        max_norm = optx.max_norm(_residual(sol.value))

        # This again is probably very sensitive to the initial conditions again, have to think about that
        def _integrate_periodic_solution(y0):
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs), solver=diffrax.Tsit5(),
                t0=self.t0, t1=self.t1, dt0=None,
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(self.t0, self.t1, self.n_time_steps)),
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
            lambda: (jnp.full((self.model.n_modes*2,), jnp.nan),
                     jnp.full((), jnp.nan))            
        )

        return y0, x_max
