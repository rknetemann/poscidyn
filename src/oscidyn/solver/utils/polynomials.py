import jax.numpy as jnp
from equinox import filter_jit

class LagrangeBasis:
    '''
    This class forms a lagrange basis of degree m. Barycentric weights are not used, because
    the lagrange polynomials are only evaluated at the collocation points. We initialize with the collocation
    points in [0,1] so we can then easily evaluate p and dp_dt at the collocation points.    
    '''

    def __init__(self, m: int, t: jnp.ndarray):
        self.m = m
        self.t = jnp.atleast_1d(t)
        self.nodes = jnp.linspace(0.0, 1.0, self.m + 1)
        
        self.Phi = self._lagrange_basis(self.t)
        self.dPhi_dt = self._derivative_lagrange_basis(self.t)

        print("Phi:", self.Phi.shape)
        print("dPhi_dt:", self.dPhi_dt.shape)

    @filter_jit
    def evaluate(self, Y: jnp.ndarray) -> jnp.ndarray:
        '''
        Evaluate the Lagrange basis polynomials and their derivatives at the given points.

        Parameters:
        Y: jnp.ndarray
            Coefficients for the Lagrange basis polynomials, shape (m+1, n)
        Returns:
        p: jnp.ndarray
            Evaluated polynomials at the collocation points, shape (m, n)
        dp_dt: jnp.ndarray
            Evaluated derivatives at the collocation points, shape (m, n)
        '''
        p = self.Phi @ Y
        dp_dt = self.dPhi_dt @ Y
        return p, dp_dt 

    @filter_jit
    def _lagrange_basis(self, t: jnp.ndarray) -> jnp.ndarray: 
        M = self.nodes.shape[0]

        diff_all = t[:, None] - self.nodes[None, :]
        P = jnp.prod(diff_all, axis=1, keepdims=True)

        dn = self.nodes[:, None] - self.nodes[None, :]
        mask = ~jnp.eye(M, dtype=bool)
        D = jnp.prod(jnp.where(mask, dn, 1.0), axis=1)

        denom = diff_all * D[None, :]
        Phi = P / denom 
        return Phi

    @filter_jit
    def _derivative_lagrange_basis(self, t: jnp.ndarray) -> jnp.ndarray:
        Phi = self._lagrange_basis(t)
        diff_all = t[:, None] - self.nodes[None, :]

        inv = 1.0 / diff_all
        sum_all = jnp.sum(inv, axis=1, keepdims=True)
        Ssum = sum_all - inv

        dPhi = Phi * Ssum
        return dPhi
    
if __name__ == "__main__":
    import numpy as np

    m = 3

    # Use midpoints between nodes to avoid division-by-zero at nodes
    nodes = jnp.linspace(0.0, 1.0, m + 1)
    t_eval = 0.5 * (nodes[:-1] + nodes[1:])  # shape (m,)

    lagrange_basis = LagrangeBasis(m, t_eval)

    Y = jnp.array([[1.0], [2.0], [3.0], [4.0]])  # shape (m+1, 1)

    p, dp_dt = lagrange_basis.evaluate(Y)

    print("t_eval:", t_eval)
    print("Lagrange Basis Evaluation:")
    print("p:", p)
    print("dp/dt:", dp_dt)

