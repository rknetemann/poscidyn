import jax
import jax.numpy as jnp
import diffrax

from .models import PhysicalModel, NonDimensionalisedModel

def solve_rhs(
        model: PhysicalModel | NonDimensionalisedModel,
        F_omega: jax.Array,
        F_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        calculate_dimless: bool = True,
    ) -> jax.Array:
            
        def _steady_state_event(self, t, state, args, **kwargs) -> jax.Array:
            del kwargs
            raise NotImplementedError("Steady state event is not implemented yet.")
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=None,
            max_steps=4096,
            y0=y0,
            throw=False,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(F_omega, F_amp),
        )
        t = sol.ts
        q = sol.ys[:, : model.N]
        v = sol.ys[:, model.N :]
            
        return t, q, v