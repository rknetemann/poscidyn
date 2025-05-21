# ───────────────────────── solver.py ──────────────────────────
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx

from .models import PhysicalModel, NonDimensionalisedModel
#from .utils.rbf_coll_solver import RBFColl

jax.config.update("jax_enable_x64", False)

def solve_rhs(
        model: PhysicalModel | NonDimensionalisedModel,
        F_omega: jax.Array,
        F_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        calculate_dimless: bool = True,
    ) -> jax.Array:
    
        period = 2 * jnp.pi / F_omega.astype(float)
           
            
        def _steady_state_event(t, state, args, **kwargs) -> jax.Array:
            del kwargs
            
            # All my steady_state check stuff
            
            return False
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=None,
            max_steps=4096,
            event=diffrax.Event(_steady_state_event),
            y0=y0,
            throw=True,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(F_omega, F_amp),
        )
        t = sol.ts
        q = sol.ys[:, : model.N]
        v = sol.ys[:, model.N :]
            
        return t, q, v