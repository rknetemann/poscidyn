from dataclasses import dataclass
import jax
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from functools import partial
from tqdm import tqdm

class ModalEOM:
    """
    N       : int             — number of modes
    c       : shape (N,)      — damping coefficients c_i
    k       : shape (N,N)     — stiffness matrix k_{ij}
    alpha   : shape (N,N,N)   — quadratic coeffs alpha_{i j k}
    gamma   : shape (N,N,N,N) — cubic    coeffs gamma_{i j k l}
    f       : shape (N,)      — forcing amplitudes f_i
    omega_d : scalar          — driving frequency
    """
    def __init__(self, omega_d: float, N: int, c: jnp.ndarray, k: jnp.ndarray, 
                 alpha: jnp.ndarray, gamma: jnp.ndarray,
                 f: jnp.ndarray):
        self.N: int = N
        self.c: jnp.ndarray = c
        self.k: jnp.ndarray = k
        self.alpha: jnp.ndarray = alpha
        self.gamma: jnp.ndarray = gamma 
        self.f: jnp.ndarray = f
        self.omega_d: float = omega_d
    
        if self.N < 1:
            raise ValueError("ModalEOM must contain at least one mode")
        if self.c.shape != (self.N,):
            raise ValueError(f"Damping coefficients must be of shape ({self.N},)")
        if self.k.shape != (self.N, self.N):
            raise ValueError(f"Stiffness matrix must be of shape ({self.N}, {self.N})")
        if self.alpha.shape != (self.N, self.N, self.N):
            raise ValueError(f"Quadratic coeffs must be of shape ({self.N}, {self.N}, {self.N})")
        if self.gamma.shape != (self.N, self.N, self.N, self.N):
            raise ValueError(f"Cubic coeffs must be of shape ({self.N}, {self.N}, {self.N}, {self.N})")
        if self.f.shape != (self.N,):
            raise ValueError(f"Forcing amplitudes must be of shape ({self.N},)")
        
    @classmethod
    def random(cls, N: int, seed: int = None) -> "ModalEOM":
        """
        Create a ModalEOM instance with random parameters using JAX.
        """
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        
        def random_array(key, shape, minval, maxval):
            key, subkey = jax.random.split(key)
            return jax.random.uniform(subkey, shape=shape, minval=minval, maxval=maxval), key

        c, key       = random_array(key, (N,), 0.0, 1.0)
        k, key       = random_array(key, (N, N), 0.0, 1.0)
        alpha, key   = random_array(key, (N, N, N), -1.0, 1.0)
        gamma, key   = random_array(key, (N, N, N, N), -1.0, 1.0)
        f, key       = random_array(key, (N,), -1.0, 1.0)
        omega_d, key = random_array(key, (), 0.0, 10.0)
        
        return cls(N=N, c=c, k=k, alpha=alpha, gamma=gamma, f=f, omega_d=float(omega_d))
    
    @classmethod
    def example(cls) -> "ModalEOM":
        """
        Create a ModalEOM instance with example parameters.
        """
        N = 2
        c = jnp.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
        k = jnp.array([[10.0, 1.0], [ 1.0,12.0]])  
        alpha = jnp.zeros((N, N, N))
        alpha = alpha.at[0].set(jnp.array([[0.0, 0.5], [0.5, 0.0]]))
        gamma = jnp.zeros((N, N, N, N))
        f = jnp.array([1.0, 0.5])
        omega_d = 3.0
        return cls(N=N, c=c, k=k, alpha=alpha, gamma=gamma, f=f, omega_d=omega_d)
    
    @classmethod
    def duffing(cls) -> "ModalEOM":
        """
        Create a ModalEOM instance with parameters for the Duffing oscillator.
        """
        N = 1

        # MATLAB parameters
        scale = 0.0
        T     = 293.0
        k_b   = 5.67e-8
        m     = 229e-12
        Q     = 30e3
        f0    = 188.1e3
        g     = 1.47e8 * (590e-9)**2
        e     = scale * g
        k1    = 320.0
        ac    = 0.59e-6

        # forcing amplitude (first of [8e-9,10e-9,12e-9,14e-9])
        F0    = 8e-9
        f_amp = F0 / k1 / ac

        # assemble arrays
        c     = jnp.array([Q])
        k     = jnp.array([[k1]])
        alpha = jnp.zeros((N, N, N))
        gamma = jnp.zeros((N, N, N, N))
        f     = jnp.array([f_amp])
        omega_d = 1.0

        print("Parameters:")
        print(f"  Damping coefficients (c): {c}")
        print(f"  Stiffness matrix (k):\n{k}")
        print(f"  Quadratic coefficients (alpha):\n{alpha}")
        print(f"  Cubic coefficients (gamma):\n{gamma}")
        print(f"  Forcing amplitudes (f): {f}")
        print(f"  Driving frequency (omega_d): {omega_d}")

        return cls(N=N, c=c, k=k, alpha=alpha, gamma=gamma, f=f, omega_d=omega_d)
    
    def eigenfrequencies(self) -> np.ndarray:
        """
        Compute the eigenfrequencies of the system.
        """
        eigvals, eigvecs = np.linalg.eig(self.k)
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        omega_n = np.sqrt(eigvals)
        return omega_n

    def steady_state_amp(self,
                     omega_d: float,
                     y0: np.ndarray,
                     t_end: float,
                     n_steps: int,
                     discard_frac: float) -> float:
        """
        Integrate the system at a given drive frequency and return
        max(|q₁|) over the steady-state portion of the response.
        """
        t_eval = np.linspace(0.0, t_end, n_steps)

        sol = solve_ivp(
            self._rhs_jax,
            (0.0, t_end),
            y0,
            t_eval=t_eval,
            args=(omega_d, self.c, self.k, self.alpha, self.gamma, self.f),
            rtol=1e-5,
            atol=1e-7,
        )

        q1 = sol.y[0]                              # first modal coordinate
        tail = q1[int(discard_frac * len(q1)):]    # discard transients
        return float(np.max(np.abs(tail)))
    
    def frequency_response(self,
                           omega_d_min: float,
                           omega_d_max: float,
                           n_omega_d: int,
                           y0: np.ndarray,
                           t_end: float,
                           n_steps: int,
                           discard_frac: float) -> tuple:
        """
        Compute the frequency response of the system.
        """
        DEVICE = jax.default_backend().upper()
        print(f"\nCalculating frequency response using {DEVICE}...")

        omega_d_grid = np.linspace(omega_d_min, omega_d_max, n_omega_d)
        amps = []
        # wrap the grid in tqdm to get a progress bar
        for ω in tqdm(omega_d_grid, desc="-> Frequency response", unit="ω"):
            amps.append(
                self.steady_state_amp(ω, y0, t_end, n_steps, discard_frac)
            )
        return omega_d_grid, np.array(amps)

    @staticmethod
    @partial(jax.jit)
    def _rhs_jax(t:float,
                 state: jnp.ndarray, 
                 omega_d:float, 
                 c: jnp.ndarray, 
                 k: jnp.ndarray, 
                 alpha: jnp.ndarray, 
                 gamma: jnp.ndarray, 
                 f: jnp.ndarray):
        """
        Compute the right-hand side (RHS) of the equations of motion using JAX for efficient computation.

        This static method calculates the acceleration and velocity of a system based on its current state,
        damping, stiffness, nonlinear coefficients, and external forcing. 
        `
        Example state: state = [q1, q2, v1, v2], where q1 and q2 are the generalized coordinates

        Args:
            t (float): The current time.
            state (jax.numpy.ndarray): The current state vector of the system, where the first half represents
                the generalized coordinates (q) and the second half represents the generalized velocities (v).
            omega_d (float): The driving frequency of the external force.
            c (jax.numpy.ndarray): The damping coefficients (1D array of size N).
            k (jax.numpy.ndarray): The stiffness matrix (2D array of shape NxN).
            alpha (jax.numpy.ndarray): The quadratic nonlinear coefficients (3D array of shape NxN).
            gamma (jax.numpy.ndarray): The cubic nonlinear coefficients (4D array of shape NxNxN).
            f (jax.numpy.ndarray): The amplitude of the external forcing (1D array of size N).

        Returns:
            jax.numpy.ndarray: The concatenated array of velocities and accelerations, representing the
            time derivative of the state vector.
        """
    
        N = c.shape[0]
        q = state[:N] # example 2 DOF: q = [q1, q2]
        v = state[N:] # example 2 DOF: v = [v1, v2]
        a_lin  = -c * v - k @ q
        a_quad = -jnp.einsum('ijk,j,k->i', alpha, q, q)
        a_cub  = -jnp.einsum('ijkl,j,k,l->i', gamma, q, q, q)
        a_forc =  f * jnp.cos(omega_d * t)
        return jnp.concatenate((v, a_lin + a_quad + a_cub + a_forc))
