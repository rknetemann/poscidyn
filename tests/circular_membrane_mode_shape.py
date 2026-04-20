from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Literal, Sequence

import numpy as np

# --- Bessel + zeros: SciPy if available, otherwise mpmath fallback ---
try:
    from scipy.special import jv as besselj  # J_m(x)
    from scipy.special import jn_zeros       # zeros of J_m
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    import mpmath as mp

    def besselj(m: int, x: np.ndarray) -> np.ndarray:
        # vectorized mpmath besselj
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x, dtype=float)
        it = np.nditer(x, flags=["multi_index"])
        while not it.finished:
            out[it.multi_index] = float(mp.besselj(m, float(it[0])))
            it.iternext()
        return out

    def jn_zeros(m: int, n: int) -> np.ndarray:
        # mpmath provides besseljzero(m, k) for k=1..n
        return np.array([float(mp.besseljzero(m, k)) for k in range(1, n + 1)], dtype=float)


AngularPart = Literal["cos", "sin", "none"]


@dataclass(frozen=True)
class MembraneMode:
    """
    Circular membrane mode indexed by (m, n) with optional angular part.

    For m=0, use angular="none".
    For m>0, you typically have a degenerate pair: angular="cos" or "sin".
    """
    m: int
    n: int
    angular: AngularPart = "cos"


@lru_cache(maxsize=None)
def _lambda_mn(m: int, n: int) -> float:
    """Return λ_{m,n} = nth zero of J_m."""
    return float(jn_zeros(m, n)[-1])


def membrane_mode_shape_vector(
    x: float | np.ndarray,
    y: float | np.ndarray,
    *,
    a: float,
    modes: Sequence[MembraneMode] | Sequence[tuple[int, int, AngularPart]],
    outside: float = np.nan,
) -> np.ndarray:
    """
    Evaluate the mode-shape vector φ(x,y) for a clamped circular membrane of radius a.

    Mode shapes (unnormalized):
        φ_{m,n}^c(r,θ) = J_m(λ_{m,n} r/a) cos(mθ)
        φ_{m,n}^s(r,θ) = J_m(λ_{m,n} r/a) sin(mθ)
        φ_{0,n}(r)     = J_0(λ_{0,n} r/a)

    Parameters
    ----------
    x, y : float or array-like
        Cartesian coordinates in the membrane plane (same units as a).
    a : float
        Membrane radius.
    modes : list of MembraneMode (or tuples (m, n, angular))
        Which modes to evaluate.
    outside : float
        Value returned for points with r > a. (np.nan by default)

    Returns
    -------
    phi : ndarray
        Shape is (*broadcast(x,y).shape, n_modes). For scalar x,y -> (n_modes,).
        These are *unnormalized* mode shape values. Use the same normalization
        convention as your modal coordinates q_i.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Broadcast base shape
    base_shape = np.broadcast(r, theta).shape

    # Normalize modes input
    mode_list: list[MembraneMode] = []
    for item in modes:
        if isinstance(item, MembraneMode):
            mode_list.append(item)
        else:
            m, n, ang = item
            mode_list.append(MembraneMode(int(m), int(n), ang))

    phi = np.empty(base_shape + (len(mode_list),), dtype=float)

    inside = r <= a
    # Avoid warnings for r/a outside; we’ll overwrite outside points anyway.
    rho = np.where(inside, r / a, 0.0)

    for k, mode in enumerate(mode_list):
        m, n, ang = mode.m, mode.n, mode.angular
        lam = _lambda_mn(m, n)
        radial = besselj(m, lam * rho)  # J_m(λ r/a)

        if m == 0 or ang == "none":
            angular = 1.0
        elif ang == "cos":
            angular = np.cos(m * theta)
        elif ang == "sin":
            angular = np.sin(m * theta)
        else:
            raise ValueError(f"Unknown angular part: {ang}")

        val = radial * angular
        # Set outside points
        if np.any(~inside):
            val = np.where(inside, val, outside)

        phi[..., k] = val

    # For scalar x,y, return (n_modes,)
    if phi.ndim == 1:
        return phi
    if phi.shape[:-1] == ():  # scalar broadcast
        return phi.reshape((len(mode_list),))
    return phi


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    a = 1.0
    modes = [
        MembraneMode(0, 1, "none"),  # (m=0,n=1) axisymmetric
        MembraneMode(1, 1, "cos"),   # one of the degenerate pair
        MembraneMode(1, 1, "sin"),   # the other degenerate pair
        MembraneMode(2, 1, "cos"),
    ]

    # Center point (middle)
    phi_center = membrane_mode_shape_vector(0.0, 0.0, a=a, modes=modes)
    print("phi(center) =", phi_center)

    # A point halfway to the edge (also sometimes called “middle” informally)
    phi_half = membrane_mode_shape_vector(a/2, 0.0, a=a, modes=modes)
    print("phi(r=a/2, theta=0) =", phi_half)