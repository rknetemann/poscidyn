# ───────────────────────── nonlinear_dynamics.py ──────────────────────────
from __future__ import annotations
import jax
import jax.numpy as jnp
import diffrax
from typing import Tuple

from oscidyn.models import PhysicalModel, NonDimensionalisedModel
from oscidyn.results import FrequencyResponse
import oscidyn.constants as const
import oscidyn

class NonlinearDynamics:
    def __init__(self, model: PhysicalModel | NonDimensionalisedModel):
        if isinstance(model, PhysicalModel):
            self.physical_model = model
            self.non_dimensionalised_model = self.physical_model.non_dimensionalise()
        elif isinstance(model, NonDimensionalisedModel):
            self.non_dimensionalised_model = model
            self.physical_model = self.non_dimensionalised_model.dimensionalise()
            #self.physical_model = None
            
    def _get_steady_state(self, q, v, discard_frac):
        n_steps = q.shape[1]
        
        q_steady = q[:, int(discard_frac * n_steps):]
        q_steady = jnp.max(jnp.abs(q_steady), axis=1)
        v_steady = v[:, int(discard_frac * n_steps):]
        v_steady = jnp.max(jnp.abs(v_steady), axis=1)
        return q_steady, v_steady
    
    def _extract_phase_lag(self, tau, q, F_omega_hat, discard_frac=0.8):
        """
        Extract the phase lag between the forcing and the displacement response
        using Fast Fourier Transform with improved noise reduction.
        
        Args:
            tau: Time array (shape: [n_cases, n_steps])
            q: Displacement array (shape: [n_cases, n_steps, N])
            F_omega_hat: Non-dimensionalized forcing frequency (shape: [n_cases])
            discard_frac: Fraction of the time series to discard for steady state
            
        Returns:
            phase_lag: Phase lag in radians (shape: [n_cases, N])
        """
        n_cases, n_steps, N = q.shape
        
        # Extract steady state portion - use more data for better frequency resolution
        start_idx = int(discard_frac * n_steps)
        q_ss = q[:, start_idx:, :]
        tau_ss = tau[:, start_idx:]
        
        # Function to calculate phase lag for one case
        def _get_phase_for_case(q_case, t_case, omega):
            # Time parameters
            dt = t_case[1] - t_case[0]
            n_samples = len(t_case)
            
            # Create Hann window for reducing spectral leakage
            window = 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n_samples) / (n_samples - 1))
            
            # Reference forcing signal (sine wave at forcing frequency)
            forcing = jnp.sin(omega * t_case) * window
            
            # Get frequency bin corresponding to forcing frequency
            freq_res = 1.0 / (n_samples * dt)
            target_bin = jnp.round(omega / (2 * jnp.pi * freq_res)).astype(jnp.int32)
            
            # Function to get phase without dynamic slicing
            def _get_phase_for_oscillator(q_osc):
                # Apply windowing
                q_windowed = q_osc * window
                
                # Apply FFT
                q_fft = jnp.fft.rfft(q_windowed)
                f_fft = jnp.fft.rfft(forcing)
                
                # Get the complex values at the target frequency bin
                # Using direct indexing (static) instead of dynamic slicing
                q_complex = q_fft[target_bin]
                f_complex = f_fft[target_bin]
                
                # Calculate phase using the complex product
                phase_diff = jnp.angle(f_complex * jnp.conj(q_complex))
                
                # Ensure phase is in [-π, π]
                phase_diff = (phase_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
                
                return phase_diff
            
            # Apply for each oscillator
            return jax.vmap(_get_phase_for_oscillator)(q_case.T)
        
        # Apply the function for each case
        phase_lag = jax.vmap(_get_phase_for_case)(q_ss, tau_ss, F_omega_hat)
        
        return phase_lag
    
    def _initial_guesses(
        self,
        F_omega_hat: jax.Array,
        F_amp_hat:  jax.Array,
        tau_end:    float,
        n_steps:    int,
        discard_frac: float,
        sweep_direction: const.Sweep = const.Sweep.FORWARD,
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

        q0_hat_n   = 50
        q0_hat     = jnp.linspace(0.01, 1.0, q0_hat_n)

        F_omega_hat_mesh, q0_hat_mesh = jnp.meshgrid(
            F_omega_hat_coarse, q0_hat, indexing="ij"
        )
        F_omega_hat_flat = F_omega_hat_mesh.ravel()
        q0_hat_flat = q0_hat_mesh.ravel()
        
        if calculate_dimless:
            model = self.non_dimensionalised_model
        else:
            model = self.physical_model

        def solve_case(f_omega_hat, q0_hat_val):
            q0 = jnp.full((N,), q0_hat_val)
            v0 = jnp.zeros((N,))
            y0 = jnp.concatenate([q0, v0])
            return oscidyn.solve_rhs(
                model, f_omega_hat, F_amp_hat, y0, tau_end * 1, n_steps,
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
        if sweep_direction == const.Sweep.FORWARD:
            sel = jnp.argmax(norm, axis=1)                         # large branch
        elif sweep_direction == const.Sweep.BACKWARD:
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
        sweep_direction: const.Sweep = const.Sweep.FORWARD,
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
            tau_end = jnp.max(tau_end) * 1.3 # 10% margin
            print(f"-> Using estimated tau_end = {tau_end:.2f} for steady state.")
        elif t_end is not None:
            tau_end = self.non_dimensionalised_model.omega_ref * t_end   
            
        if n_steps is None:
            n_steps = 2000 
            print(f"-> Using default n_steps = {n_steps}.")     
            
        if discard_frac is None:
            discard_frac = 0.8
            
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
        
        if calculate_dimless:
            model = self.non_dimensionalised_model
        else:
            model = self.physical_model  
                    
        def solve_rhs(F_omega_hat, F_amp_hat, y0_hat):
            return oscidyn.solve_rhs(model, F_omega_hat, F_amp_hat, y0_hat, tau_end, n_steps, steady_state=False, calculate_dimless=calculate_dimless)

        print("-> Solving for steady state:")
        tau, q, v = jax.vmap(solve_rhs, in_axes=(0, None, 0))(F_omega_hat_grid, F_amp_hat_grid, y0_hat_grid)
        q_st, v_st = self._get_steady_state(q, v, discard_frac)
        
        phase = self._extract_phase_lag(tau, q, F_omega_hat_grid, discard_frac)
        #phase = None
        
        q_st_total = jnp.sum(q_st, axis=1)
        
        frequency_response = FrequencyResponse(
            excitation_type="forward",
            excitation_frequency=F_omega_hat_grid,
            excitation_amplitude=F_amp_hat_grid,
            total_response=q_st_total,
            mode_response=q_st,
            initial_guesses=y0_hat_grid
        )

        return frequency_response

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
        
        if calculate_dimless:
            model = self.non_dimensionalised_model
        else:
            model = self.physical_model
            
        # Call _solve_rhs directly, no vmap needed for a single phase portrait
        tau, q, v = oscidyn.solve_rhs(
            model=model,
            F_omega=F_omega_hat, # Passed as F_omega to _solve_rhs, interpreted as F_omega_hat by model
            F_amp=F_amp_hat,     # Passed as F_amp to _solve_rhs, interpreted as F_amp_hat by model
            y0=y0_hat,
            t_end=tau_end,
            n_steps=n_steps,
            t_steady_state_check=None,
            calculate_dimless=calculate_dimless,
        )
        
        return tau, q, v

    def time_response(
        self,
        F_omega: jax.Array = None,
        F_omega_hat: jax.Array = None,
        F_amp: jax.Array = None,
        F_amp_hat: jax.Array = None,
        y0: jax.Array = None,
        y0_hat: jax.Array = None,
        t_end: float = None,
        tau_end: float = None,
        n_steps: int = None,
        calculate_dimless: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Compute the time response of the system for a given set of parameters.
        
        Args:
            F_omega: Dimensional forcing frequency
            F_omega_hat: Non-dimensional forcing frequency
            F_amp: Dimensional forcing amplitude
            F_amp_hat: Non-dimensional forcing amplitude
            y0: Dimensional initial condition [q0, v0]
            y0_hat: Non-dimensional initial condition [q0_hat, v0_hat]
            t_end: End time (dimensional)
            tau_end: End time (non-dimensional)
            n_steps: Number of time steps
            calculate_dimless: Whether to calculate in non-dimensional form
            
        Returns:
            tau/t: Time array (non-dimensional/dimensional)
            q: Displacement array (shape: [n_steps+1, N])
            v: Velocity array (shape: [n_steps+1, N])
        """
        print("\n Calculating time response:")
        
        if calculate_dimless:
            model = self.non_dimensionalised_model
        else:
            model = self.physical_model
        
        N = model.N
        
        # Handle frequency parameters
        if F_omega is not None and F_omega_hat is not None:
            raise ValueError("Either F_omega or F_omega_hat must be provided, not both.")
        elif F_omega_hat is None and F_omega is None:
            F_omega_hat = model.F_omega_hat
        elif F_omega is not None and calculate_dimless:
            F_omega_hat = F_omega / self.non_dimensionalised_model.omega_ref
        elif F_omega_hat is not None and not calculate_dimless:
            F_omega = F_omega_hat * self.non_dimensionalised_model.omega_ref
        
        # Handle amplitude parameters
        if F_amp is not None and F_amp_hat is not None:
            raise ValueError("Either F_amp or F_amp_hat must be provided, not both.")
        elif F_amp_hat is None and F_amp is None:
            F_amp_hat = model.F_amp_hat
        elif F_amp is not None and calculate_dimless:
            F_amp_hat = F_amp / (self.non_dimensionalised_model.m * 
                               self.non_dimensionalised_model.omega_ref**2 * 
                               self.non_dimensionalised_model.x_ref)
        elif F_amp_hat is not None and not calculate_dimless:
            F_amp = F_amp_hat * (self.non_dimensionalised_model.m * 
                               self.non_dimensionalised_model.omega_ref**2 * 
                               self.non_dimensionalised_model.x_ref)
        
        # Handle time parameters
        if tau_end is not None and t_end is not None:
            raise ValueError("Either t_end or tau_end must be provided, not both.")
        elif tau_end is None and t_end is None:
            tau_end = 300.0  # Default non-dimensional time
        elif t_end is not None and calculate_dimless:
            tau_end = self.non_dimensionalised_model.omega_ref * t_end
        elif tau_end is not None and not calculate_dimless:
            t_end = tau_end / self.non_dimensionalised_model.omega_ref
        
        if n_steps is None:
            n_steps = 2000  # Default number of steps
        
        # Handle initial conditions
        if y0_hat is not None and y0 is not None:
            raise ValueError("Either y0 or y0_hat must be provided, not both.")
        elif y0_hat is None and y0 is None:
            y0_hat = jnp.zeros(2 * N)  # Default zero initial conditions
        elif y0 is not None and calculate_dimless:
            q0 = y0[:N] / self.non_dimensionalised_model.x_ref
            v0 = y0[N:] / (
                self.non_dimensionalised_model.x_ref
                * self.non_dimensionalised_model.omega_ref
            )
            y0_hat = jnp.concatenate([q0, v0])
        elif y0_hat is not None and not calculate_dimless:
            q0 = y0_hat[:N] * self.non_dimensionalised_model.x_ref
            v0 = y0_hat[N:] * (
                self.non_dimensionalised_model.x_ref
                * self.non_dimensionalised_model.omega_ref
            )
            y0 = jnp.concatenate([q0, v0])
        
        # Use appropriate parameters based on calculate_dimless
        F_param = F_omega_hat if calculate_dimless else F_omega
        F_amp_param = F_amp_hat if calculate_dimless else F_amp
        y0_param = y0_hat if calculate_dimless else y0
        t_param = tau_end if calculate_dimless else t_end
        
        # Solve the system
        print(f"-> Solving time response with {n_steps} steps...")
        tau, q, v = oscidyn.solve_rhs(
            model=model,
            F_omega=F_param,
            F_amp=F_amp_param,
            y0=y0_param,
            t_end=t_param,
            n_steps=n_steps,
            calculate_dimless=calculate_dimless,
        )
        
        return tau, q, v