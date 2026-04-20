import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, combinations


def hb_solutions_at_w(m, c, k, alpha, F, w, rtol_imag=1e-8):
    """
    Return all real positive harmonic-balance solutions at frequency w:
    arrays As (amplitudes) and stabs (bool stability) of same length (1..3).
    """
    zeta = c / (2 * m)
    w0_sq = k / m

    a = (3 / 4) * alpha
    b = (k - m * w * w)
    d = (c * w)

    # cubic in x = A^2:
    # (a^2)x^3 + (2ab)x^2 + (b^2 + d^2)x - F^2 = 0
    p3 = a * a
    p2 = 2 * a * b
    p1 = b * b + d * d
    p0 = -F * F

    roots = np.roots([p3, p2, p1, p0])

    xs = []
    for r in roots:
        if abs(r.imag) > rtol_imag:
            continue
        x = float(r.real)
        if x > 0:
            xs.append(x)

    if not xs:
        return np.array([]), np.array([], dtype=bool)

    xs = np.array(sorted(xs))
    As = np.sqrt(xs)

    stabs = []
    for x, A in zip(xs, As):
        denom = np.sqrt((b + a * x) ** 2 + d ** 2)
        if denom == 0:
            stabs.append(False)
            continue

        cosphi = (b + a * x) / denom
        sinphi = d / denom

        u = A * cosphi
        v = A * sinphi
        r2 = u * u + v * v

        sigma = (w0_sq - w * w) + (3 * alpha / (4 * m)) * r2

        # Jacobian of averaged slow-flow in (u,v)
        ds_du = (3 * alpha / (2 * m)) * u
        ds_dv = (3 * alpha / (2 * m)) * v

        a11 = -zeta + (v / (2 * w)) * ds_du
        a12 = (sigma / (2 * w)) + (v / (2 * w)) * ds_dv
        a21 = -(sigma / (2 * w)) - (u / (2 * w)) * ds_du
        a22 = -zeta - (u / (2 * w)) * ds_dv

        eig = np.linalg.eigvals(np.array([[a11, a12], [a21, a22]], dtype=float))
        stabs.append(np.all(np.real(eig) < 0))

    return As, np.array(stabs, dtype=bool)


def assign_by_continuity(prevA, As, birth_pen=0.2, death_pen=0.5):
    """
    Assign current solution amplitudes As (len 0..3) to 3 branch slots
    to minimize mismatch to previous branch amplitudes prevA (len 3).

    Returns:
        newA (len 3, float with NaN),
        idx_map: list of length len(As) giving assigned branch index for each solution
    """
    # If no solutions, everything dies
    if len(As) == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float), []

    branch_indices = [0, 1, 2]
    best_cost = np.inf
    best_newA = None
    best_map = None

    # Choose which branch slots are occupied this step (size = len(As))
    for chosen in combinations(branch_indices, len(As)):
        # For those chosen slots, permute assignment order
        for perm in permutations(chosen, len(As)):
            # perm[j] = branch index for As[j]
            cost = 0.0
            newA = np.array([np.nan, np.nan, np.nan], dtype=float)

            used = set(perm)
            # assignment cost
            for j, bi in enumerate(perm):
                newA[bi] = As[j]
                if np.isfinite(prevA[bi]):
                    cost += abs(As[j] - prevA[bi])
                else:
                    cost += birth_pen

            # penalty for dropping an existing branch that was previously finite
            for bi in branch_indices:
                if np.isfinite(prevA[bi]) and (bi not in used):
                    cost += death_pen

            if cost < best_cost:
                best_cost = cost
                best_newA = newA
                best_map = list(perm)

    return best_newA, best_map


def duffing_tracked_branches(m, c, k, alpha, F, w_grid,
                            rtol_imag=1e-8, birth_pen=0.2, death_pen=0.5):
    """
    Compute up to 3 tracked HB branches across w_grid.
    Returns:
        Abranches: list of 3 arrays
        Sbranches: list of 3 bool arrays
    """
    Abranches = [np.full_like(w_grid, np.nan, dtype=float) for _ in range(3)]
    Sbranches = [np.full_like(w_grid, False, dtype=bool) for _ in range(3)]

    prevA = np.array([np.nan, np.nan, np.nan], dtype=float)

    for i, w in enumerate(w_grid):
        As, stabs = hb_solutions_at_w(m, c, k, alpha, F, w, rtol_imag=rtol_imag)

        # Sort solutions by amplitude (for deterministic behavior)
        if len(As) > 0:
            order = np.argsort(As)
            As = As[order]
            stabs = stabs[order]

        newA, mapping = assign_by_continuity(prevA, As, birth_pen=birth_pen, death_pen=death_pen)

        # Fill outputs
        for bi in range(3):
            if np.isfinite(newA[bi]):
                Abranches[bi][i] = newA[bi]

        # Set stability for assigned points
        for sol_j, bi in enumerate(mapping):
            Sbranches[bi][i] = bool(stabs[sol_j])

        prevA = newA

    return Abranches, Sbranches


def plot_duffing(w_grid, Abranches, Sbranches):
    plt.figure()

    for A_b, S_b in zip(Abranches, Sbranches):
        mask = np.isfinite(A_b)

        ms = mask & S_b
        if np.any(ms):
            plt.plot(w_grid[ms], A_b[ms], color="black", linestyle="-", linewidth=2.0)

        mu = mask & (~S_b)
        if np.any(mu):
            plt.plot(w_grid[mu], A_b[mu], color="black", linestyle=":", linewidth=2.5)

    plt.xlabel(r"$\omega$")
    plt.ylabel(r"$A$")
    plt.title("Frequency response")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # ---- parameters (edit) ----
    m = 1.0
    c = 0.04
    k = 1.0
    alpha = 0.1   # >0 hardening, <0 softening
    F = 0.2

    w = np.linspace(0.6, 1.6, 1200)

    Abranches, Sbranches = duffing_tracked_branches(
        m, c, k, alpha, F, w,
        birth_pen=0.15, death_pen=0.35
    )

    plot_duffing(w, Abranches, Sbranches)
