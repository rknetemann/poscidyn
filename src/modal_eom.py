from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from numbalsoda import lsoda_sig, lsoda
from numba import njit, cfunc, float64, int32
from numba.experimental import jitclass

@cfunc(lsoda_sig)
def rhs_numba(t, state, d_state, params):
    # params: omega_d, N, c, k, alpha, gamma, f
    omega_d = params[0]
    N = params[1]
    c = params[2]
    k = params[3]
    alpha = params[4]
    gamma = params[5]
    f = params[6]
    
    q = state[:N]
    v = state[N:]

    a_lin  = -c * v - k @ q
    a_quad = -np.einsum('ijk,j,k->i', alpha, q, q)   # ok only if your Numba version supports einsum
    a_cub  = -np.einsum('ijkl,j,k,l->i', gamma, q, q, q)
    a_forc =  f * np.cos(omega_d * t)

    return np.concatenate((v, a_lin + a_quad + a_cub + a_forc))

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
    def __init__(self, N: int, c: np.ndarray, k: np.ndarray, 
                 alpha: np.ndarray, gamma: np.ndarray,
                 f: np.ndarray, omega_d: float):
        self.N: int = N
        self.c: np.ndarray = c
        self.k: np.ndarray = k
        self.alpha: np.ndarray = alpha
        self.gamma: np.ndarray = gamma 
        self.f: np.ndarray = f
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
        Create a ModalEOM instance with random parameters.
        """
        rng = np.random.default_rng(seed)
        c       = rng.uniform(0.0, 1.0, size=N)
        k       = rng.uniform(0.0, 1.0, size=(N, N))
        alpha   = rng.uniform(-1.0, 1.0, size=(N, N, N))
        gamma   = rng.uniform(-1.0, 1.0, size=(N, N, N, N))
        f       = rng.uniform(-1.0, 1.0, size=N)
        omega_d = float(rng.uniform(0.0, 10.0))
        return cls(N=N, c=c, k=k, alpha=alpha, gamma=gamma, f=f, omega_d=omega_d)
    
    @classmethod
    def example(cls) -> "ModalEOM":
        """
        Create a ModalEOM instance with example parameters.
        """
        N = 2
        c = np.array([2.0 * 0.01 * 5.0, 2.0 * 0.02 * 8.0])
        k = np.array([[10.0, 1.0], [ 1.0,12.0]])  
        alpha = np.zeros((N, N, N))
        alpha[0] = np.array([[0.0, 0.5], [0.5, 0.0]])
        gamma = np.zeros((N, N, N, N))
        f = np.array([1.0, 0.5])
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
        c     = np.array([Q])
        k     = np.array([[k1]])
        alpha = np.zeros((N, N, N))
        gamma = np.zeros((N, N, N, N))
        f     = np.array([f_amp])
        omega_d = 1.0

        return cls(N=N, c=c, k=k, alpha=alpha, gamma=gamma, f=f, omega_d=omega_d)
    
    def eigenfrequencies(self) -> np.ndarray:
        """
        Compute the eigenfrequencies of the system.
        """
        # solve the eigenvalue problem
        eigvals, eigvecs = np.linalg.eig(self.k)
        # sort eigenvalues and eigenvectors
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        # compute natural frequencies
        omega_n = np.sqrt(eigvals)
        return omega_n
    
    def rhs(self, t: float, state: np.ndarray, omega_d: float) -> np.ndarray:
        """
        Compute the time derivative of state = [q, v].

        Parameters
        ----------
        t : float
            Current time.
        state : ndarray, shape (2*N,)
            Concatenated [q (N,), v (N,)].

        Returns
        -------
        dstate_dt : ndarray, shape (2*N,)
            [v, a], where a includes linear, quadratic, cubic, and forcing terms.
        """        
        return rhs_numba(t, state, omega_d,
                     self.N, self.c, self.k,
                     self.alpha, self.gamma, self.f)
    
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

        sol = lsoda(rhs_numba,
                    u0=y0,
                    t_eval=t_eval,
                    data=np.array([omega_d, self.N, self.c, self.k, self.alpha, self.gamma, self.f]),             # pass ω_d into rhs
                    rtol=1e-5, atol=1e-7,)          # swap for "Radau"/"BDF" if stiff

        q1 = sol.y[0]                              # first modal coordinate
        tail = q1[int(discard_frac * len(q1)):]    # discard transients
        return float(np.max(np.abs(tail)))
    
    def frequency_response(self,
                        omega_d_min: float = 0.0,
                        omega_d_max: float = 500.0,
                        n_omega_d: int = 50,
                        y0: np.ndarray = None,
                        t_end: float = 250.0,
                        n_steps: int = 500,
                        discard_frac: float = 0.8) -> tuple:
        """
        Compute the frequency response of the system.
        """
        if y0 is None:
            y0 = np.zeros(2 * self.N)

        omega_d_grid = np.linspace(omega_d_min, omega_d_max, n_omega_d)
        amps = np.array([
            self.steady_state_amp(omega_d, y0, t_end, n_steps, discard_frac)
            for omega_d in omega_d_grid
        ])
        return omega_d_grid, amps
