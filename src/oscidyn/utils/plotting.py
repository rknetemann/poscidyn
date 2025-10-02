import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from matplotlib import cm, colors

from .. import constants as const

BIF_LABELS = {0: "none", 1: "SN", 2: "PD", 3: "NS"}

def _truncate_cmap(name="Greys", lo=0.3, hi=1.0, N=256):
    base = cm.get_cmap(name, N)
    return colors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(lo, hi, N))
    )

def plot_branch_exploration(
    coarse_drive_freq_mesh,
    coarse_drive_amp_mesh,
    results,
    *,
    stable_size=60,
    unstable_size=16,
    tol_inside=1e-4,
    annotate_bifurcations=True,
    cmap_lo=0.15,     # <- raise this if you want darker minimum
    cmap_hi=0.90,
):
    # flatten grids
    freq_vals = jnp.asarray(coarse_drive_freq_mesh).ravel()
    amp_vals  = jnp.asarray(coarse_drive_amp_mesh).ravel()
    n_sim     = int(freq_vals.size)

    # helpers
    def _flat(name, default=None):
        if name in results and results[name] is not None:
            return jnp.asarray(results[name]).reshape(-1)
        return default

    x_max = _flat("x_max", default=jnp.full((n_sim,), jnp.nan))

    if "stable" in results:
        stable = _flat("stable")
    elif "rho_max" in results:
        rho_max = _flat("rho_max")
        stable = jnp.isfinite(rho_max) & (rho_max <= (1.0 - tol_inside))
    elif "mu" in results:
        mu = jnp.asarray(results["mu"]).reshape(n_sim, -1)
        rho = jnp.nanmax(jnp.abs(mu), axis=1)
        stable = jnp.isfinite(rho) & (rho <= (1.0 - tol_inside))
    else:
        stable = jnp.zeros((n_sim,), dtype=bool)

    sizes = jnp.where(stable, stable_size, unstable_size)
    bif = _flat("bifurcation", default=jnp.zeros((n_sim,), dtype=jnp.int32))

    # truncated grayscale so lowest value is NOT white
    cmap = _truncate_cmap("Greys", cmap_lo, cmap_hi)

    # ensure vmin < vmax (e.g., if only one amplitude)
    vmin = float(amp_vals.min())
    vmax = float(amp_vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        eps = 1e-12
        vmin -= eps
        vmax += eps

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        np.asarray(freq_vals),
        np.asarray(x_max),
        c=np.asarray(amp_vals),
        s=np.asarray(sizes),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        alpha=0.9,
    )
    fig.colorbar(sc, ax=ax, label="Driving amplitude")

    # stability legend
    stable_dot   = ax.scatter([], [], s=stable_size,   c="k", alpha=0.9, label="stable")
    unstable_dot = ax.scatter([], [], s=unstable_size, c="k", alpha=0.9, label="unstable")
    legend_handles = [stable_dot, unstable_dot]

    # optional bifurcation overlays
    if annotate_bifurcations:
        sn_mask = np.asarray(bif == 1)
        pd_mask = np.asarray(bif == 2)
        ns_mask = np.asarray(bif == 3)

        if sn_mask.any():
            h_sn = ax.scatter(
                np.asarray(freq_vals)[sn_mask],
                np.asarray(x_max)[sn_mask],
                marker="o", facecolors="none", edgecolors="C3", s=80, linewidths=1.5, label="SN",
            ); legend_handles.append(h_sn)

        if pd_mask.any():
            h_pd = ax.scatter(
                np.asarray(freq_vals)[pd_mask],
                np.asarray(x_max)[pd_mask],
                marker="s", facecolors="none", edgecolors="C0", s=80, linewidths=1.5, label="PD",
            ); legend_handles.append(h_pd)

        if ns_mask.any():
            h_ns = ax.scatter(
                np.asarray(freq_vals)[ns_mask],
                np.asarray(x_max)[ns_mask],
                marker="^", facecolors="none", edgecolors="C2", s=80, linewidths=1.5, label="NS",
            ); legend_handles.append(h_ns)

    if legend_handles:
        ax.legend(handles=legend_handles, title="Floquet")

    ax.set_xlabel("Driving frequency")
    ax.set_ylabel("Steady-state displacement amplitude")
    ax.set_title("Branch Exploration")
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.savefig("branch_exploration.png", dpi=300)
    plt.show()
