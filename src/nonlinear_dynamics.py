# ───────────────────────── nonlinear_dynamics.py ──────────────────────────
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import optimistix

from models import PhysicalModel, NonDimensionalisedModel

jax.config.update("jax_enable_x64", False)

class NonlinearDynamics:
    def __init__(self, model: PhysicalModel | NonDimensionalisedModel):
        if isinstance(model, PhysicalModel):
            self.physical_model = model
            self.non_dimensionalised_model = self.physical_model.non_dimensionalise()
        elif isinstance(model, NonDimensionalisedModel):
            self.non_dimensionalised_model = model
            #self.physical_model = self.non_dimensionalised_model.dimensionalise()
            self.physical_model = None
            
    def _get_steady_state(self, q, v, discard_frac):
        n_steps = q.shape[1]
        
        q_steady = q[:, int(discard_frac * n_steps):]
        q_steady = jnp.max(jnp.abs(q_steady), axis=1)
        v_steady = v[:, int(discard_frac * n_steps):]
        v_steady = jnp.max(jnp.abs(v_steady), axis=1)
        return q_steady, v_steady

    # --------------------------------------------------- internal solver
    def _solve_rhs(
        self,
        F_omega: jax.Array,
        F_amp: jax.Array,
        y0: jax.Array,
        t_end: float,
        n_steps: int,
        calculate_dimless: bool = True,
    ) -> jax.Array:
        
        if calculate_dimless:
            model = self.non_dimensionalised_model
        else:
            model = self.physical_model
            
        def _steady_state_event(self, t, state, args, **kwargs) -> jax.Array:
            del kwargs
            raise NotImplementedError("Steady state event is not implemented yet.")
        
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=None,
            max_steps=400096,
            y0=y0,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(F_omega, F_amp),
        )
        t = sol.ts
        q = sol.ys[:, : model.N]
        v = sol.ys[:, model.N :]
            
        return t, q, v
    
    # --------------------------------------------------- public wrappers
    def frequency_response(
        self,
        F_omega: jax.Array = None,
        F_omega_hat: jax.Array = None,
        F_amp : jax.Array = None,
        F_amp_hat: jax.Array = None,
        y0: jax.Array = None,
        y0_hat: jax.Array = None,
        t_end: float = None,
        tau_end: float = None,
        n_steps: int = None,
        discard_frac: float = None,
        calculate_dimless: bool = True,
    ) -> tuple:
        
        if F_omega and F_omega_hat:
            raise ValueError("Either F_omega or F_omega_hat must be provided, not both.")
        
        if F_omega_hat is None:
            if F_omega is None:
                F_omega_hat_max = 3.0 * self.non_dimensionalised_model.omega_0_hat[-1]
                F_omega_hat = jnp.linspace(0, F_omega_hat_max, 400)
            else:
                F_omega_hat = F_omega / self.non_dimensionalised_model.omega_ref
            
        if F_amp_hat is None:
            if F_amp is None:
                if calculate_dimless:
                    F_amp_hat = self.non_dimensionalised_model.F_amp_hat
                else:
                    F_amp = self.physical_model.F_amp
                    F_amp_hat = F_amp / (self.m * self.non_dimensionalised_model.omega_ref**2 * self.non_dimensionalised_model.x_ref)

        if y0_hat is not None and y0 is not None:
            raise ValueError("Either y0 or y0_hat must be provided, not both.")
        
        if calculate_dimless:
            N = self.non_dimensionalised_model.N
        else:
            N = self.physical_model.N
            
        if y0_hat is None:
            if y0 is None:
                y0 = jnp.zeros(2 * N)
                print("-> Using default initial conditions y0 = 0.")
            
            q0 = y0[:N] / self.non_dimensionalised_model.x_ref
            v0 = y0[N:] / (self.non_dimensionalised_model.x_ref * self.non_dimensionalised_model.omega_ref)
            y0_hat = jnp.concatenate([q0, v0])
            
        if tau_end and t_end:
            raise ValueError("Either t_end or tau_end must be provided, not both.")
            
        if tau_end is None:
            if t_end is None:
                damping_ratio = 1 / (2 * self.non_dimensionalised_model.Q)
                tau_end = jnp.max(3.9 / (self.non_dimensionalised_model.omega_0_hat * damping_ratio))
                tau_end = jnp.max(tau_end) * 1.1 # 10% margin
                print(f"-> Using estimated tau_end = {tau_end:.2f} for steady state.")
            else:
                tau_end = self.non_dimensionalised_model.omega_ref * t_end    
            
        if n_steps is None:
            n_steps = 2000 
            print(f"-> Using default n_steps = {n_steps}.")       
            
        if discard_frac is None:
            T_hat = 2*jnp.pi / self.non_dimensionalised_model.omega_0_hat
            steady_state_frac = 5 * T_hat / tau_end # 5 periods are considered for steady state
            discard_frac = jnp.max(1 - steady_state_frac)
            print(f"-> Using estimated discard_frac = {discard_frac:.2f}.")

        def solve_rhs(F_omega_hat, F_amp_hat):
            return self._solve_rhs(F_omega_hat, F_amp_hat, y0_hat, tau_end, n_steps)

        tau, q, v = jax.vmap(solve_rhs, in_axes=(0, None))(F_omega_hat, F_amp_hat)
        q_st, v_st = self._get_steady_state(q, v, discard_frac)
        
        q_st_total = jnp.sum(q_st, axis=1)
        
        return F_omega_hat, q_st, q_st_total, v_st

    def phase_portrait(
        self,
        F_omega_hat: jax.Array = None,
        F_amp_hat: jax.Array = None,
        y0_hat: jax.Array = None,
        tau_end: float = None,
        n_steps: int = None,
        calculate_dimless: bool = True, # Currently, only non-dimensionalised is robustly supported for simulation
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        
        if not calculate_dimless:
            # Physical model simulation path might need review of PhysicalModel._build_rhs
            raise NotImplementedError("Phase portrait for physical model directly is not fully verified.")

        model_to_use = self.non_dimensionalised_model
        N = model_to_use.N

        if F_omega_hat is None:
            F_omega_hat = model_to_use.F_omega_hat
        
        if F_amp_hat is None:
            F_amp_hat = model_to_use.F_amp_hat

        if y0_hat is None:
            y0_hat = jnp.zeros(2 * N)
            
        if tau_end is None:
            tau_end = 200.0 # Default non-dimensional time
            
        if n_steps is None:
            n_steps = 4000 # Default number of steps

        # Call _solve_rhs directly, no vmap needed for a single phase portrait
        tau, q, v = self._solve_rhs(
            F_omega=F_omega_hat, # Passed as F_omega to _solve_rhs, interpreted as F_omega_hat by model
            F_amp=F_amp_hat,     # Passed as F_amp to _solve_rhs, interpreted as F_amp_hat by model
            y0=y0_hat,
            t_end=tau_end,
            n_steps=n_steps,
            calculate_dimless=calculate_dimless,
        )
        
        return tau, q, v