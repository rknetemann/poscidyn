# solver.py
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx

from .models import AbstractModel

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')

# import os
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
# )

class AbstractSolver:
    def __init__(self, n_steps: int = 2000, max_steps: int = 4096):
        self.n_steps = n_steps
        self.max_steps = max_steps

class StandardSolver(AbstractSolver):
    def __init__(self, t_end: float, n_steps: int = 2000, max_steps: int = 4096):
        super().__init__(n_steps)
        self.t_end = t_end
        self.max_steps = max_steps

    def solve_rhs(self, model: AbstractModel, driving_frequency: jax.Array, driving_amplitude: jax.Array, initial_condition: jax.Array) -> jax.Array:
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=self.t_end,
            dt0=None,
            max_steps=self.max_steps,
            y0=initial_condition,
            throw=True,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, self.t_end, self.n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
            args=(driving_frequency, driving_amplitude),
        )

        time = sol.ts
        displacements = sol.ys[:, : model.n_modes]
        velocities = sol.ys[:, model.n_modes :]
        
        return time, displacements, velocities

class SteadyStateSolver(AbstractSolver):
    def __init__(self):
        pass

    def solve_rhs(self, model: AbstractModel, F_omega: jax.Array, F_amp: jax.Array, y0: jax.Array, t_end: float, n_steps: int) -> jax.Array:
        raise NotImplementedError("SteadyStateSolver is not implemented yet.")