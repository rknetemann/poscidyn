from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array, Float, PyTree

from .abstract_oscillator import AbstractOscillator

class Nonlinear(AbstractOscillator):
    def __init__(self, omega_0: Array = None, Q: Array = None, a: Array = None, b: Array = None, n_modes: int = None):
        if n_modes is not None:
            if omega_0 is None:
                omega_0 = np.zeros(n_modes)
            if Q is None:
                Q = np.zeros(n_modes)
            if a is None:
                a = np.zeros((n_modes, n_modes, n_modes))
            if b is None:
                b = np.zeros((n_modes, n_modes, n_modes, n_modes))

            if omega_0.shape != (n_modes,):
                raise ValueError("omega_0 must have shape (n_modes,)")
            if Q.shape != (n_modes,):
                raise ValueError("Q must have shape (n_modes,)")
            if a.shape != (n_modes, n_modes, n_modes):
                raise ValueError("a must have shape (n_modes, n_modes, n_modes)")
            if b.shape != (n_modes, n_modes, n_modes, n_modes):
                raise ValueError("b must have shape (n_modes, n_modes, n_modes, n_modes)")
        else:
            if n_modes is None:
                if omega_0 is None or Q is None or a is None or b is None:
                    raise ValueError("n_modes must be specified if any of the arrays are None")
                n_modes = omega_0.shape[0]
                
        self.n_modes = n_modes     
        self.omega_0 = omega_0
        self.Q = Q
        self.a = a
        self.b = b

    def f_i(self, t: Float, y: Array, args: PyTree, omega_ref: float = 1.0, x_ref: float = 1.0) -> Array:
        q, dq_dt   = jnp.split(y, 2)

        ETA = 0.01

        damping_term = (self.omega_0/omega_ref) * 1/self.Q * dq_dt
        nonlinear_damping_term = ETA * q**2 * dq_dt
        linear_stiffness_term = (1/omega_ref**2) * self.omega_0**2 * q
        quadratic_stiffness_term = (x_ref / omega_ref**2) * jnp.einsum("ijk,j,k->i", self.a, q, q)
        cubic_stiffness_term = (x_ref**2 / omega_ref**2) * jnp.einsum("ijkl,j,k,l->i", self.b, q, q, q) # Shape: (n_modes,)

        d2q_dt2 = (
            - damping_term
            - nonlinear_damping_term
            - linear_stiffness_term
            - quadratic_stiffness_term
            - cubic_stiffness_term
        ) 
        return d2q_dt2

    # Not yet used, but for future shooting and collocation methods we will need it
    def f_i_y(self, t: Float, y: Array, args: PyTree) -> Array:
        q, dq_dt = jnp.split(y, 2)

        zero_block = jnp.zeros((self.n_modes, self.n_modes))
        identity_block = jnp.eye(self.n_modes)
        A_bottom_left = -jnp.diag(self.g2 + 3 * self.g3 * q**2)
        A_bottom_right = -jnp.diag(self.g1)

        A = jnp.block([[zero_block, identity_block],
                       [A_bottom_left, A_bottom_right]])
        return A

    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:
        '''driving_frequency
        Calculates the settling time for a given Q-factor and driving frequency.
        Equation from Eq.5.10b Vibrations 2nd edition by Balakumar Balachandran | Edward B. Magrab
        '''
        #driving_frequency = 1.0
        t_steady_y = jnp.max(-2 * jnp.max(self.Q) * jnp.log(ss_tol * jnp.sqrt(1 - 1 / (4 * jnp.max(self.Q)**2)) / (driving_frequency))).reshape(())

        return t_steady_y

    @property
    def n_dof(self) -> int:
        return self.n_modes
    
    def __repr__(self):
        Q_terms = ", ".join([f"Q[{i}]={float(v):.6f}" for i, v in enumerate(self.Q)])
        omega_0_terms = ", ".join([f"omega_0[{i}]={float(v):.6f}" for i, v in enumerate(self.omega_0)])

        a_indices, a_value = jnp.where(self.a != 0.0), self.a[jnp.where(self.a != 0.0)]
        a_terms = [f"a[{i[0]},{i[1]},{i[2]}]={float(v):.6f}" for i, v in zip(zip(*a_indices), a_value)]
        if len(a_terms) > 20:
            a_terms = a_terms[:20] + ["... (truncated)"]
        a_str = f", {', '.join(a_terms)}" if a_terms else ""

        b_indices, b_value = jnp.where(self.b != 0.0), self.b[jnp.where(self.b != 0.0)]
        b_terms = [f"b[{i[0]},{i[1]},{i[2]},{i[3]}]={float(v):.6f}" for i, v in zip(zip(*b_indices), b_value)]
        if len(b_terms) > 20:
            b_terms = b_terms[:20] + ["... (truncated)"]
        b_str = f", {', '.join(b_terms)}" if b_terms else ""

        return (f"Nonlinear(n_modes={self.n_modes}, "
                f"{Q_terms}, {omega_0_terms}{a_str}{b_str})")
