# ───────────────────────── solver.py ──────────────────────────
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx

from .models import PhysicalModel, NonDimensionalisedModel

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')

import os
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
)

def solve_rhs(
        model: PhysicalModel | NonDimensionalisedModel,
        excitation_frequency: jax.Array,
        excitation_amplitude: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int = 4096,
        steady_state_termination: bool = False
    ) -> jax.Array:
    
        if steady_state_termination:
            raise NotImplementedError(
                "Steady state termination is not implemented yet."
            )
        else:
            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(model.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=0.0,
                t1=t_end,
                dt0=None,
                max_steps=400096,
                y0=y0,
                throw=False,
                progress_meter=diffrax.TqdmProgressMeter(),
                saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                args=(excitation_frequency, excitation_amplitude),
            )
        t = sol.ts
        q = sol.ys[:, : model.N]
        v = sol.ys[:, model.N :]

        return t, q, v

# This function solves the RHS for each excitation frequency and amplitude in a batch manner.
vmap_y0   = jax.vmap(
    solve_rhs,
    in_axes=(None, None, None, 0,    None, None, None)
)
vmap_amp  = jax.vmap(
    vmap_y0,
    in_axes=(None, None, 0,    None, None, None, None)
)
solve_rhs_batched = jax.vmap(
    vmap_amp,
    in_axes=(None, 0,    None, None, None, None, None)
)


