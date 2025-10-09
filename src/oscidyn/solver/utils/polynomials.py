import jax
import jax.numpy as jnp
from equinox import filter_jit

class LagrangeBasis:
    def __init__(self, nodes: jnp.ndarray):
        self.tau = jnp.asarray(nodes)
        self.K = self.tau.size - 1
        self.w = self._barycentric_weights(self.tau)
        self._D = self._differentiation_matrix(self.tau, self.w)

    @staticmethod
    def _barycentric_weights(x: jnp.ndarray) -> jnp.ndarray:
        diff = x[:, None] - x[None, :]
        diff = diff + jnp.eye(x.size)
        return 1.0 / jnp.prod(diff, axis=1)

    @staticmethod
    def _differentiation_matrix(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        X = x[:, None] - x[None, :]
        off = ~jnp.eye(x.size, dtype=bool)
        Dij = jnp.where(off, (w[None, :] / (w[:, None] * X)), 0.0)
        Dii = -jnp.sum(Dij, axis=1)
        return Dij + jnp.diag(Dii)

    @staticmethod
    def _basis_matrix(x: jnp.ndarray, nodes: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        x = x[..., None]
        X = x - nodes[None, :]
        close = jnp.isclose(X, 0.0)
        any_hit = jnp.any(close, axis=1, keepdims=True)
        S = w[None, :] / X
        B = S / jnp.sum(S, axis=1, keepdims=True)
        hit = close / jnp.sum(close, axis=1, keepdims=True)
        return jnp.where(any_hit, hit, B)

    @staticmethod
    def _interp_and_deriv(x: jnp.ndarray, nodes: jnp.ndarray, w: jnp.ndarray, y: jnp.ndarray, D_nodes: jnp.ndarray):
        y2 = y if y.ndim == 2 else y[:, None]
        xcol = x[:, None]
        X = xcol - nodes[None, :]
        close = jnp.isclose(X, 0.0)
        any_hit = jnp.any(close, axis=1, keepdims=True)
        S = w[None, :] / X
        den = jnp.sum(S, axis=1, keepdims=True)
        num = S @ y2
        p = num / den
        Sprime = -w[None, :] / (X * X)
        den_p = jnp.sum(Sprime, axis=1, keepdims=True)
        num_p = Sprime @ y2
        dp = (num_p - p * den_p) / den
        hit_idx = jnp.argmax(close, axis=1)
        Dy = D_nodes @ y2
        y_hit = y2[hit_idx, :]
        dy_hit = Dy[hit_idx, :]
        y_out = jnp.where(any_hit, y_hit, p)
        dy_out = jnp.where(any_hit, dy_hit, dp)
        if y.ndim == 1:
            y_out = y_out.squeeze(-1)
            dy_out = dy_out.squeeze(-1)
        return y_out, dy_out

    @filter_jit
    def basis_matrix(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._basis_matrix(x, self.tau, self.w)

    @filter_jit
    def basis_derivative_matrix(self, x: jnp.ndarray) -> jnp.ndarray:
        I = jnp.eye(self.K + 1)
        _, dB = self._interp_and_deriv(x, self.tau, self.w, I, self._D)
        return dB

    @filter_jit
    def evaluate(self, Y: jnp.ndarray, x: jnp.ndarray | None = None) -> jnp.ndarray:
        if x is None:
            x = self.tau
        y, _ = self._interp_and_deriv(x, self.tau, self.w, Y, self._D)
        return y

    @filter_jit
    def derivative(self, Y: jnp.ndarray, x: jnp.ndarray | None = None) -> jnp.ndarray:
        if x is None:
            x = self.tau
        _, dy = self._interp_and_deriv(x, self.tau, self.w, Y, self._D)
        return dy

    @filter_jit
    def differentiation_matrix(self) -> jnp.ndarray:
        return self._D


if __name__ == "__main__":
    from jax import config
    config.update("jax_enable_x64", True)
    import numpy as np
    from numpy.polynomial.legendre import leggauss

    f  = lambda x: jnp.sin(2 * jnp.pi * x)
    df = lambda x: 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)

    # ---- configuration (like AUTO) ----
    K = 10          # polynomial degree (cubic)
    NCOL = K + 1   # number of collocation points per interval
    NTST = 20      # number of intervals
    h = 1.0 / NTST # element length

    # ---- Gauss–Legendre nodes on [0,1] ----
    z, _ = leggauss(NCOL)             # nodes in [-1,1]
    tau_local = 0.5 * (z + 1.0)       # map to [0,1]
    lb_local = LagrangeBasis(tau_local)

    # ---- assemble global differentiation matrix ----
    # Each element is handled separately; we combine results piecewise
    x_all = []
    f_num = []
    df_num = []
    df_true = []

    for i in range(NTST):
        a, b = i * h, (i + 1) * h
        # map local nodes [0,1] -> [a,b]
        x_elem = a + h * tau_local
        y_elem = f(x_elem)

        # local derivative in physical space: (1/h) * D * y
        D_local = (1.0 / h) * lb_local.differentiation_matrix()
        dy_elem = D_local @ y_elem

        x_all.append(x_elem)
        f_num.append(y_elem)
        df_num.append(dy_elem)
        df_true.append(df(x_elem))

    x_all = jnp.concatenate(x_all)
    f_num = jnp.concatenate(f_num)
    df_num = jnp.concatenate(df_num)
    df_true = jnp.concatenate(df_true)

    err = jnp.abs(df_num - df_true)
    print(f"Degree {K} with {NTST} elements")
    print(f"Max derivative error: {err.max():.3e}")
