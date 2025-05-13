# ───────────────────────── nonlinear_dynamics.py ──────────────────────────
from __future__ import annotations
import jax
import jax.numpy as jnp
import diffrax
from typing import Tuple

from .models import PhysicalModel, NonDimensionalisedModel
from .constants import Sweep, Y0_HAT_COARSE_N, F_OMEGA_HAT_COARSE_N

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
    
    def _initial_guesses(
        self,
        F_omega_hat: jax.Array,
        F_amp_hat:  jax.Array,
        tau_end:    float,
        n_steps:    int,
        discard_frac: float,
        sweep_direction: Sweep = Sweep.FORWARD,
        calculate_dimless: bool = True,
    ) -> Tuple[jax.Array, jax.Array]:
        
        def _pick_inphase_sample(tau, q, v, ω_F):
            phase = (ω_F * tau) % (2 * jnp.pi)
            k = jnp.argmin(jnp.abs(phase))
            return q[k], v[k]

        N = self.non_dimensionalised_model.N
        F_omega_hat_fine = F_omega_hat                   
        
        F_omega_hat_n   = 50
        F_omega_hat_min = jnp.min(F_omega_hat_fine)
        F_omega_hat_max = jnp.max(F_omega_hat_fine)
        F_omega_hat_coarse = jnp.linspace(                
            F_omega_hat_min, F_omega_hat_max, F_omega_hat_n
        )

        q0_hat_n   = 30
        q0_hat     = jnp.linspace(0.0, 1.0, q0_hat_n)

        F_omega_hat_mesh, q0_hat_mesh = jnp.meshgrid(
            F_omega_hat_coarse, q0_hat, indexing="ij"
        )
        F_omega_hat_flat = F_omega_hat_mesh.ravel()
        q0_hat_flat = q0_hat_mesh.ravel()

        def solve_case(f_omega_hat, q0_hat_val):
            q0 = jnp.full((N,), q0_hat_val)
            v0 = jnp.zeros((N,))
            y0 = jnp.concatenate([q0, v0])
            return self._solve_rhs(
                f_omega_hat, F_amp_hat, y0, tau_end * 2, n_steps,
                calculate_dimless=calculate_dimless,
            )

        tau_flat, q_flat, v_flat = jax.vmap(solve_case)(
            F_omega_hat_flat, q0_hat_flat
        )

        T0 = int(discard_frac * n_steps)
        tau_cut = tau_flat[:, T0:]        # shape (n_cases, n_keep)
        q_cut   = q_flat[:,  T0:, :]      # (n_cases, n_keep, N)
        v_cut   = v_flat[:,  T0:, :]

        pick = lambda t,q,v,ω: _pick_inphase_sample(t, q, v, ω)
        q_steady, v_steady = jax.vmap(pick)(tau_cut, q_cut, v_cut, F_omega_hat_flat)

        q_steady_state = q_steady.reshape(F_omega_hat_n, q0_hat_n, N)
        v_steady_state = v_steady.reshape(F_omega_hat_n, q0_hat_n, N)

        norm = jnp.linalg.norm(q_steady_state, axis=-1)            # (freq , n_init)
        if sweep_direction == Sweep.FORWARD:
            sel = jnp.argmax(norm, axis=1)                         # large branch
        elif sweep_direction == Sweep.BACKWARD:
            sel = jnp.argmin(norm, axis=1)                         # small branch

        rows = jnp.arange(F_omega_hat_n)
        q0_coarse = q_steady_state[rows, sel, :]                   # (freq , N)
        v0_coarse = v_steady_state[rows, sel, :]                   # (freq , N)

        interp = lambda y: jnp.interp(
            F_omega_hat_fine,           # x-length target
            F_omega_hat_coarse,         # 50-length source x
            y                           # 50-length values
        )
       
        q0 = jax.vmap(interp, in_axes=1, out_axes=1)(q0_coarse)  # (x,N)
        v0 = jax.vmap(interp, in_axes=1, out_axes=1)(v0_coarse)
            
        y0 = jnp.concatenate([q0, v0], axis=-1)            # (x,2N)
        
        return y0
        
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
        F_omega_grid: jax.Array = None,
        F_omega_hat_grid: jax.Array = None,
        F_amp_grid : jax.Array = None,
        F_amp_hat_grid: jax.Array = None,
        y0: jax.Array = None,
        y0_hat: jax.Array = None,
        t_end: float = None,
        tau_end: float = None,
        n_steps: int = None,
        discard_frac: float = None,
        calculate_dimless: bool = True,
        sweep_direction: Sweep = Sweep.FORWARD,
    ) -> tuple:
        print("\n Calculating frequency response:")
        
        if calculate_dimless:
            N = self.non_dimensionalised_model.N
        else:
            N = self.physical_model.N
        
        if F_omega_grid is not None and F_omega_hat_grid is not None:
            raise ValueError("Either F_omega or F_omega_hat must be provided, not both.")
        elif F_omega_hat_grid is None and F_omega_grid is None:
            F_omega_hat_min = 0.0
            F_omega_hat_max = 3.0 * self.non_dimensionalised_model.omega_0_hat[-1]
            F_omega_hat_grid = jnp.linspace(F_omega_hat_min, F_omega_hat_max, 400)
        elif F_omega_grid is not None:
            F_omega_hat_grid = F_omega_grid / self.non_dimensionalised_model.omega_ref
        if F_amp_grid is not None and F_amp_hat_grid is not None:
            raise ValueError("Either F_amp or F_amp_hat must be provided, not both.")
        elif F_amp_hat_grid is None and F_amp_grid is None:
            F_amp_hat_grid = self.non_dimensionalised_model.F_amp_hat
        elif F_amp_grid is not None:
            F_amp_hat_grid = F_amp_grid / (self.m * self.non_dimensionalised_model.omega_ref**2 * self.non_dimensionalised_model.x_ref)
                
        if tau_end is not None and t_end is not None:
            raise ValueError("Either t_end or tau_end must be provided, not both.")  
        elif tau_end is None and t_end is None:
            damping_ratio = 1 / (2 * self.non_dimensionalised_model.Q)
            tau_end = jnp.max(3.9 / (self.non_dimensionalised_model.omega_0_hat * damping_ratio))
            tau_end = jnp.max(tau_end) * 1.1 # 10% margin
            print(f"-> Using estimated tau_end = {tau_end:.2f} for steady state.")
        elif t_end is not None:
            tau_end = self.non_dimensionalised_model.omega_ref * t_end   
            
        if n_steps is None:
            n_steps = 2000 
            print(f"-> Using default n_steps = {n_steps}.")     
            
        if discard_frac is None:
            T_hat = 2*jnp.pi / self.non_dimensionalised_model.omega_0_hat
            steady_state_frac = 5 * T_hat / tau_end # 5 periods are considered for steady state
            discard_frac = jnp.max(1 - steady_state_frac)
            print(f"-> Using estimated discard_frac = {discard_frac:.2f}.")
            
        if y0_hat is not None and y0 is not None:
            raise ValueError("Either y0 or y0_hat must be provided, not both.")
        elif y0_hat is None and y0 is None:
            print("-> Calculating initial guesses y0:")
            y0_hat_grid = self._initial_guesses(
                F_omega_hat=F_omega_hat_grid,
                F_amp_hat=F_amp_hat_grid,
                tau_end=tau_end,
                n_steps=n_steps,
                discard_frac=discard_frac,
                sweep_direction=sweep_direction or ForwardSweep(),
                calculate_dimless=calculate_dimless,
            )     
        elif y0 is not None:
            q0 = y0[:N] / self.non_dimensionalised_model.x_ref
            v0 = y0[N:] / (
                self.non_dimensionalised_model.x_ref
                * self.non_dimensionalised_model.omega_ref
            )
            y0_hat = jnp.concatenate([q0, v0])
            y0_hat_grid = jnp.broadcast_to(y0_hat, (F_omega_hat_grid.size, 2 * N))   
                    
        def solve_rhs(F_omega_hat, F_amp_hat, y0_hat):
            return self._solve_rhs(F_omega_hat, F_amp_hat, y0_hat, tau_end, n_steps)

        print("-> Solving for steady state:")
        tau, q, v = jax.vmap(solve_rhs, in_axes=(0, None, 0))(F_omega_hat_grid, F_amp_hat_grid, y0_hat_grid)
        q_st, v_st = self._get_steady_state(q, v, discard_frac)
        
        q_st_total = jnp.sum(q_st, axis=1)
        
        return F_omega_hat_grid, q_st, q_st_total, v_st, y0_hat_grid

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