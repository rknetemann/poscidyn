from __future__ import annotations
"""
Radial Basis‑Function collocation integrator for **Diffrax**
===========================================================
Implementation based on

> Elgohary, T., Virgin, L., Gopalakrishnan, J. & Dowell, E. H. (2014)
> *A Simple, Fast, and Accurate Time‑Integrator for Strongly Nonlinear Dynamical Systems*.

This file defines an implicit, fixed‑step solver `RBFColl` that plugs into the
Diffrax ecosystem. It is JIT‑compatible and differentiable, yet all heavy
linear‑algebra work is pre‑computed with **NumPy** at import‑time to avoid GPU
solver dependencies.

Usage
-----
```python
from diffrax import diffeqsolve, ODETerm
from rbf_coll_solver import RBFColl

solver = RBFColl()            # or pass your own hyper‑params
sol = diffeqsolve(term, solver, ...)
```
"""

from typing import Any, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from diffrax._solution import RESULTS

Array = jax.Array
PyTree = Any

# -----------------------------------------------------------------------------
# Helper utilities (NumPy, CPU‑side) ------------------------------------------
# -----------------------------------------------------------------------------


def legendre_gauss_lobatto_nodes(n: int) -> np.ndarray:
    """Return *n* Legendre–Gauss–Lobatto nodes in the interval ``[-1, 1]``."""
    if n < 2:
        raise ValueError("Need at least two nodes for LGL quadrature.")
    if n == 2:
        return np.array([-1.0, 1.0], dtype=float)
    interior, _ = np.polynomial.legendre.leggauss(n - 2)
    nodes = np.concatenate(([-1.0], interior, [1.0]))
    return nodes.astype(float)


def _rbf_matrices(nodes: np.ndarray, c: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return evaluation matrix ``Φ`` and differentiation matrix ``D`` (NumPy)."""
    r = nodes[:, None] - nodes[None, :]
    phi = np.sqrt(1.0 + (c * r) ** 2, dtype=float)
    dphi = (c**2 * r) / np.sqrt(1.0 + (c * r) ** 2, dtype=float)
    D = dphi @ np.linalg.solve(phi, np.eye(phi.shape[0], dtype=float))
    return phi, D


# -----------------------------------------------------------------------------
# Solver ----------------------------------------------------------------------
# -----------------------------------------------------------------------------


class RBFColl(dfx.AbstractImplicitSolver, eqx.Module):
    """Implicit RBF‑collocation solver (fixed step, spectral accuracy)."""

    # Public hyper‑parameters ----------------------------------------------
    n_collocation: int = 7
    shape_param: float = 2.5

    # Static attributes (non‑differentiable) --------------------------------
    root_finder: optx.AbstractRootFinder = eqx.static_field()
    root_find_max_steps: int = eqx.static_field(default=20)
    _nodes_unit: np.ndarray = eqx.static_field()
    _D_unit: np.ndarray = eqx.static_field()

    # Diffrax metadata -----------------------------------------------------
    term_structure = dfx.ODETerm
    interpolation_cls = dfx.LocalLinearInterpolation

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        n_collocation: int = 7,
        shape_param: float = 2.5,
        root_finder: optx.AbstractRootFinder | None = None,
        root_find_max_steps: int = 20,
    ) -> None:
        object.__setattr__(self, "n_collocation", n_collocation)
        object.__setattr__(self, "shape_param", shape_param)
        if root_finder is None:
            root_finder = optx.Newton(rtol=1e-10, atol=1e-12)
        object.__setattr__(self, "root_finder", root_finder)
        object.__setattr__(self, "root_find_max_steps", root_find_max_steps)

        # Pre‑compute collocation data on CPU; stored as NumPy arrays so the
        # GPU driver is never touched at import‑time.
        nodes_unit = legendre_gauss_lobatto_nodes(n_collocation)
        _, D_unit = _rbf_matrices(nodes_unit, shape_param)
        object.__setattr__(self, "_nodes_unit", nodes_unit)
        object.__setattr__(self, "_D_unit", D_unit)

    # ------------------------------------------------------------------
    # Diffrax API
    # ------------------------------------------------------------------
    def order(self, terms):  # noqa: D401
        return None  # Spectral in *n*; no simple integer order

    def init(self, terms, t0, t1, y0, args):  # noqa: D401
        return None  # Stateless per step

    # Vector‑field helper --------------------------------------------------
    def func(self, terms, t, y, args):
        try:
            return terms.vf(t, y, args)
        except AttributeError:
            return terms.func(t, y, args)

    # ------------------------------------------------------------------
    # Core algorithm – single integrator step
    # ------------------------------------------------------------------
    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        dt = t1 - t0
        nodes = (jnp.asarray(self._nodes_unit) + 1.0) * (0.5 * dt) + t0
        D = jnp.asarray(self._D_unit) / dt

        # Flatten state once; produce an unravel function for reconstruction.
        y0_flat, unravel = jax.flatten_util.ravel_pytree(y0)
        dim = y0_flat.size

        def ravel_tree(tree):
            return jax.flatten_util.ravel_pytree(tree)[0]

        # ---------------------- Residual of collocation equations --------
        def _pack(inner: Array) -> Array:
            return jnp.concatenate([y0_flat, inner.reshape(-1)])

        def residual(inner: Array, *_):
            ys_flat = _pack(inner).reshape(self.n_collocation, dim)
            ys_tree = jax.vmap(unravel)(ys_flat)
            vf_nodes = jax.vmap(lambda tt, yy: self.func(terms, tt, yy, args))(nodes, ys_tree)
            vf_flat = jax.vmap(ravel_tree)(vf_nodes)
            res = (D @ ys_flat - vf_flat)[1:]  # skip first row
            return res.reshape(-1)

        # ---------------------- Initial guess (Forward‑Euler string) -----
        dt_euler = dt / (self.n_collocation - 1)
        curr_t, curr_y = t0, y0
        inner0_chunks: list[Array] = []
        for _ in range(1, self.n_collocation):  # nodes 1 … n‑1
            dy = self.func(terms, curr_t, curr_y, args)
            curr_y = jax.tree_util.tree_map(lambda v, dv: v + dt_euler * dv, curr_y, dy)
            inner0_chunks.append(ravel_tree(curr_y))
            curr_t += dt_euler
        inner0 = jnp.concatenate(inner0_chunks)

        # ---------------------- Non‑linear solve -------------------------
        sol = optx.root_find(
            residual,
            self.root_finder,
            inner0,
            max_steps=self.root_find_max_steps,
            throw=False,
        )
        inner = sol.value

        # ---------------------- Extract endpoint ------------------------
        ys_flat = _pack(inner).reshape(self.n_collocation, dim)
        y1 = unravel(ys_flat[-1])

        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, solver_state, RESULTS.successful


# -----------------------------------------------------------------------------
# Quick self‑test (Duffing oscillator) ----------------------------------------
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import jax.numpy as jnp

    def duffing(t, y, args):
        x, v = y
        delta, alpha, omega = args
        return jnp.array([
            v,
            -delta * v - x - alpha * x**3 + 0.3 * jnp.cos(omega * t),
        ])

    term = dfx.ODETerm(duffing)
    solver = RBFColl(n_collocation=7, shape_param=2.5)

    t0, t1, dt = 0.0, 20.0, 0.1  # shorter demo
    y0 = jnp.array([1.0, 0.0])
    args = (0.2, 0.01, 1.0)

    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        args=args,
        saveat=dfx.SaveAt(t1=True),
        throw=False,
    )
