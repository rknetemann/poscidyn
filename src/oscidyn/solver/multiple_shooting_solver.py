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
        self.verbose = verbose

        self.model: AbstractModel = None

        self.multistart.verbose = self.verbose

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

        y0, _, _ = self._calc_periodic_solution(drive_freq, drive_amp, y0_guess)

        def _solve_one_period(y0):               
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Kvaerno5(),
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
        
        ts = jnp.linspace(self.t0, self.t1, self.m_segments + 1)
    
        s0 = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(self._rhs), solver=diffrax.Kvaerno5(),
            t0=self.t0, t1=self.t1, dt0=None,
            y0=y0_guess,
            saveat=diffrax.SaveAt(ts=ts),
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
                solver=diffrax.Kvaerno5(),
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

        # def _residual(s, _args=None):               # s: (m_segments, n_modes * 2)
        #     def _one(carry, i):
        #         sk  = s[i]
        #         sk1 = s[(i + 1) % self.m_segments]    # wrap for periodicity
        #         Phi = _integrate_segment(sk, ts[i], ts[i + 1])  # (n_modes * 2,)
        #         r = Phi - sk1
        #         return carry, r

        #     _, Rs = jax.lax.scan(_one, None, jnp.arange(self.m_segments))  # Rs: (m_segments, n_modes * 2)
        #     return Rs.reshape((-1,))                         # (m_segments * n_modes * 2,)
        
        def _residual(s, _args=None): # s: (m_segments + 1, n_modes * 2)
            t0 = ts[:-1]
            t1 = ts[1:]
            s_next = jnp.roll(s, shift=-1, axis=0)

            Phi = jax.vmap(_integrate_segment, in_axes=(0, 0, 0))(s, t0, t1)  # (m, n)
            R = Phi - s_next                                                  # (m, n)
            return R.reshape(-1)                                              # (m*n,)

        solver = optx.LevenbergMarquardt(rtol=1e-6, atol=1e-8, norm=optx.rms_norm) # for debugging: verbose=frozenset({"step", "accepted", "loss", "step_size"})
        sol = optx.least_squares(_residual, solver, y0=s0, options={"jac": "bwd"}, max_steps=self.max_shooting_iterations, throw=False) 

        # This again is probably very sensitive to the initial conditions again, have to think about that
        def _postprocess_success(s_star):
            y0_periodic = s_star[0]

            # fine trajectory to get x_max
            sol_traj = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs), solver=diffrax.Kvaerno5(),
                t0=self.t0, t1=self.t1, dt0=None,
                y0=y0_periodic,
                saveat=diffrax.SaveAt(ts=jnp.linspace(self.t0, self.t1, self.n_time_steps)),
                throw=False,
                max_steps=self.max_steps,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),
            )
            xs  = sol_traj.ys[:, :self.model.n_modes]
            x_max = jnp.max(jnp.abs(xs))

            # Floquet multipliers from monodromy (via jacobian of flow)
            mu = self._compute_floquet_multipliers(y0_periodic, driving_amplitude, driving_frequency)

            # classify
            stable, bif_code, rho_max = self._classify_multipliers(mu)

            return {
                "y0": y0_periodic,                 # (2*n_modes,)
                "x_max": x_max,                    # scalar
                "mu": mu,                          # (2*n_modes,) complex
                "rho_max": rho_max,                # scalar
                "stable": stable,                  # bool
                "bifurcation": bif_code,           # int32
            }
            
        def _postprocess_fail(_):
            return {
                "y0": jnp.full((self.model.n_modes*2,), jnp.nan),
                "x_max": jnp.nan,
                "mu": jnp.full((self.model.n_modes*2,), jnp.nan + 0j),
                "rho_max": jnp.nan,
                "stable": False,
                "bifurcation": jnp.array(0, dtype=jnp.int32),
            }
        
        out = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda: _postprocess_success(sol.value),
            lambda: _postprocess_fail(None)
        )

        return out

    @filter_jit
    def _compute_floquet_multipliers(self, y0, driving_amplitude, driving_frequency):
        """
        Compute Floquet multipliers using automatic differentiation.
        
        The monodromy matrix M is the Jacobian of the flow map Φ(T, y0) with respect to y0.
        The Floquet multipliers are the eigenvalues of M.
        """
        
        # Define the flow map for one period
        def flow_map(y):
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(self._rhs),
                solver=diffrax.Kvaerno5(),
                t0=self.t0, t1=self.t1, dt0=None,
                adjoint=diffrax.ForwardMode(),
                y0=y,
                saveat=diffrax.SaveAt(t1=True),
                throw=False,
                max_steps=self.max_steps,
                progress_meter=diffrax.NoProgressMeter(),
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                args=(driving_amplitude, driving_frequency),
            )
            return sol.ys.squeeze(0)
        
        # Compute the monodromy matrix using JAX's automatic differentiation
        monodromy_matrix = jax.jacfwd(flow_map)(y0)
        
        # Compute eigenvalues (Floquet multipliers)
        # Note: For real systems, eigenvalues might be complex
        eigenvalues, _ = jnp.linalg.eig(monodromy_matrix)
        
        return eigenvalues
    
    def _classify_multipliers(self, mu,
                            tol_inside=1e-6,   # margin inside unit circle for "stable"
                            tol_sn=1e-3,       # |mu - 1| small
                            tol_pd=1e-3,       # |mu + 1| small
                            tol_ns_r=1e-3,     # ||mu|-1| small
                            tol_ns_ang=0.2):   # angle not near 0 or pi
        # mu: (n,) complex
        rho = jnp.abs(mu)
        rho_max = jnp.max(rho)

        # stable if strictly inside by a little margin
        stable = rho_max <= (1.0 - tol_inside)

        # saddle-node: multiplier near +1
        sn = jnp.any(jnp.abs(mu - (1.0 + 0j)) < tol_sn)

        # period doubling: multiplier near -1
        pd = jnp.any(jnp.abs(mu + (1.0 + 0j)) < tol_pd)

        # Neimark–Sacker: complex pair near unit circle with nontrivial angle
        ang = jnp.angle(mu)
        complex_mask = (jnp.abs(jnp.imag(mu)) > 1e-9)
        near_circle  = jnp.abs(rho - 1.0) < tol_ns_r
        ang_nontrivial = (jnp.abs(jnp.mod(ang, jnp.pi)) > tol_ns_ang)
        ns = jnp.any(complex_mask & near_circle & ang_nontrivial)

        # 0:none/unknown  1:SN  2:PD  3:NS
        bif_code = jnp.where(sn, 1,
                    jnp.where(pd, 2,
                        jnp.where(ns, 3, 0))).astype(jnp.int32)

        return stable, bif_code, rho_max
    
    @filter_jit
    def _rhs(self, tau, y, args):
        _drive_amp, drive_freq  = args

        t = tau / drive_freq
        dydt = self.model.f(t, y, args) / drive_freq

        return dydt
