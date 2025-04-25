from dataclasses import dataclass
import numpy as np

@dataclass
class NormalForm:
    """
    N       : int             — number of modes
    c       : shape (N,)      — damping coefficients c_i
    k       : shape (N,N)     — stiffness matrix k_{ij}
    alpha   : shape (N,N,N)   — quadratic coeffs alpha_{i j k}
    gamma   : shape (N,N,N,N) — cubic    coeffs gamma_{i j k l}
    f       : shape (N,)      — forcing amplitudes f_i
    omega_d : scalar          — driving frequency
    """
    N: int
    c: np.ndarray
    k: np.ndarray
    alpha: np.ndarray
    gamma: np.ndarray
    f: np.ndarray
    omega_d: float

    def __post_init__(self):
        if self.N < 1:
            raise ValueError("NormalForm must contain at least one mode")
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
    def random(cls, N: int, seed: int = None) -> "NormalForm":
        """
        Create a NormalForm instance with random parameters.
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
    def example(cls) -> "NormalForm":
        """
        Create a NormalForm instance with example parameters.
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

    def rhs(self, t: float, state: np.ndarray, omega_d: float = None) -> np.ndarray:
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
        if omega_d is None:
            omega_d = self.omega_d
        
        # unpack
        q = state[:self.N]
        v = state[self.N:]

        # linear part:  -c_i v_i   - sum_j k_{ij} q_j
        #a_lin = -self.c * v - self.k @ q
        a_lin = - self.k @ q # no damping term

        # quadratic:  - sum_{j,k} alpha_{i j k} q_j q_k
        a_quad = -np.einsum('ijk,j,k->i', self.alpha, q, q)

        # cubic:      - sum_{j,k,l} gamma_{i j k l} q_j q_k q_l
        a_cub  = -np.einsum('ijkl,j,k,l->i', self.gamma, q, q, q)

        # forcing:    f_i * cos(omega_d * t)
        a_forc = self.f * np.cos(omega_d * t)

        # total acceleration
        a = a_lin + a_quad + a_cub + a_forc

        # return [dq/dt, dv/dt]
        return np.concatenate([v, a])
    