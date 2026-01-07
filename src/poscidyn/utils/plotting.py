import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from matplotlib import cm, colors

from .. import constants as const

BIF_LABELS = {0: "none", 1: "SN", 2: "PD", 3: "NS"}

def get_data():

    # Load nonlinear sweep data
    file = f"tests/chris/16_40_15-LDV_100-nonlinear_sweep2_data.txt"
    data = np.loadtxt(file, delimiter=",", skiprows=1)  # Skip the header row
    freq = data[:, 0]
    mag = data[:, 1]
    phase = data[:, 2]

    def measurement2meter(freq, mag, setup, *args):
        if setup == "ENIGMA with correction":
            # ENIGMA - using transduction
            transduction = args[0]
            delta = mag / transduction * 1e-9
            gamma = 4*np.pi / 632.8e-9  # 4 pi / lambda_red
            lin_error = 1/8*delta*2 * gamma*2
            return delta * (1 - lin_error)

        elif setup == "ENIGMA":
            # ENIGMA - using transduction
            transduction = args[0]
            return mag / transduction * 1e-9

        elif setup == "Polytec":
            # Polytec - using LDV settings
            LDV = args[0]
            return 2 * mag * LDV / (2 * np.pi * freq)

        else:
            print(f'Setup should be "ENIGMA" or "ENIGMA with correction" or "Polytec", not "{setup}"')

    settings = "LDV_100"
    setup = "Polytec"; calibration = float(settings.split('_')[-1]) * 1e-3
    total_points = 11; picked_point = 2

    x_ref = 1e-8
    omega_ref = 207.708e3

    freq_sweep = freq
    modeshape_coeff1 = np.sin((picked_point - 1) / (total_points - 1) * np.pi)
    mag_sweep = measurement2meter(freq_sweep, mag, setup, calibration) / modeshape_coeff1

    freq_sweep = freq_sweep / omega_ref
    mag_sweep = mag_sweep / x_ref

    return freq_sweep, mag_sweep

def _truncate_cmap(name="Greys", lo=0.3, hi=1.0, N=256):
    base = cm.get_cmap(name, N)
    return colors.LinearSegmentedColormap.from_list(
        f"{name}_trunc", base(np.linspace(lo, hi, N))
    )

def _envelope_by_bins(f, y, *, mask=None, nbins=300, agg="max"):
    """
    Build a single-valued 'envelope' y_env(f_env) from scattered (f, y) by binning.
    agg: 'max' (upper branch) or 'median' (central tendency).
    """
    f = np.asarray(f).ravel()
    y = np.asarray(y).ravel()
    if mask is None:
        mask = np.isfinite(f) & np.isfinite(y)
    else:
        mask = mask & np.isfinite(f) & np.isfinite(y)

    if mask.sum() < 3:
        return np.array([]), np.array([])

    fmin, fmax = float(np.min(f[mask])), float(np.max(f[mask]))
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmin == fmax:
        return np.array([]), np.array([])

    edges = np.linspace(fmin, fmax, nbins + 1)
    idx = np.digitize(f[mask], edges) - 1  # bin indices in [0, nbins-1]
    f_env = []
    y_env = []
    for b in range(nbins):
        sel = idx == b
        if not np.any(sel):
            continue
        fbin = f[mask][sel]
        ybin = y[mask][sel]
        if agg == "median":
            yv = float(np.median(ybin))
        else:
            yv = float(np.max(ybin))
        f_env.append(0.5 * (edges[b] + edges[b + 1]))
        y_env.append(yv)
    return np.asarray(f_env), np.asarray(y_env)

def _weighted_stats(x, y, power=2.0):
    """
    Resonance-weighted mean and std of x using weights ~ y^power (y assumed >= 0).
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan, np.nan
    w = np.maximum(y[m], 0.0) ** power
    if w.sum() == 0:
        return np.nan, np.nan
    mu = np.sum(w * x[m]) / np.sum(w)
    var = np.sum(w * (x[m] - mu) ** 2) / np.sum(w)
    return mu, np.sqrt(var)

def _align_measurement_to_sim(f_meas, x_meas, f_env, y_env, *, power=2.0):
    """
    Find linear frequency map f' = s * f_meas + t and amplitude scale c
    so that x_meas aligns to y_env on the overlapping domain.
    """
    # Clean + sort
    f_meas = np.asarray(f_meas).ravel()
    x_meas = np.asarray(x_meas).ravel()
    m = np.isfinite(f_meas) & np.isfinite(x_meas)
    f_meas, x_meas = f_meas[m], x_meas[m]
    order = np.argsort(f_meas)
    f_meas, x_meas = f_meas[order], x_meas[order]

    f_env = np.asarray(f_env).ravel()
    y_env = np.asarray(y_env).ravel()
    m2 = np.isfinite(f_env) & np.isfinite(y_env)
    f_env, y_env = f_env[m2], y_env[m2]
    order2 = np.argsort(f_env)
    f_env, y_env = f_env[order2], y_env[order2]

    if f_meas.size < 3 or f_env.size < 3:
        # Fallback: identity mapping, unit amplitude
        return 1.0, 0.0, 1.0

    # Resonance-weighted centroid+width alignment (robust for unimodal resonance)
    mu_m, sd_m = _weighted_stats(f_meas, x_meas, power=power)
    mu_s, sd_s = _weighted_stats(f_env, y_env, power=power)
    if not np.isfinite(mu_m) or not np.isfinite(mu_s) or sd_m <= 0 or not np.isfinite(sd_s):
        s, t = 1.0, 0.0
    else:
        s = sd_s / sd_m if sd_m > 0 else 1.0
        t = mu_s - s * mu_m

    f_meas_mapped = s * f_meas + t

    # Compute amplitude scale c on the overlapping region by least squares
    # Interpolate measurement -> env grid
    within = (f_env >= f_meas_mapped.min()) & (f_env <= f_meas_mapped.max())
    if np.any(within):
        x_on_env = np.interp(f_env[within], f_meas_mapped, x_meas)
        y_tgt = y_env[within]
        finite = np.isfinite(x_on_env) & np.isfinite(y_tgt)
        if np.any(finite):
            denom = float(np.sum(x_on_env[finite] ** 2))
            c = float(np.sum(x_on_env[finite] * y_tgt[finite]) / denom) if denom > 0 else 1.0
        else:
            c = 1.0
    else:
        c = 1.0

    return s, t, c

def plot_branch_exploration(
    coarse_drive_freq_mesh,
    coarse_drive_amp_mesh,
    results,
    *,
    stable_size=60,
    unstable_size=16,
    tol_inside=1e-4,
    annotate_bifurcations=False,
    cmap_lo=0.15,
    cmap_hi=0.90,
    backbone=None,
    backbone_n=300,
    backbone_kwargs=None,
    title="Branch Exploration",
    overlay_get_data=True,
    align_measurement=True,     # <— auto-align frequency & amplitude so curves line up
    show_interpolated=False,     # <— also show markers at sim grid (one-to-one comparison)
    env_bins=300,               # <— envelope resolution for alignment/comparison
    env_agg="max",              # 'max' = upper branch, 'median' = central tendency
):
    # --- flatten grids
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

    # --- colormap
    cmap = _truncate_cmap("Greys", cmap_lo, cmap_hi)

    # --- color scale bounds
    vmin = float(amp_vals.min())
    vmax = float(amp_vals.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        eps = 1e-12
        vmin -= eps
        vmax += eps

    # --- figure
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

    # --- backbone (optional)
    if backbone is not None:
        f0   = float(backbone["f0"])
        beta = float(backbone["beta"])
        A_min = float(np.nanmin(np.asarray(x_max)))
        A_max = float(np.nanmax(np.asarray(x_max)))
        if not np.isfinite(A_min) or not np.isfinite(A_max):
            A_min, A_max = 0.0, 1.0
        A_min = max(0.0, A_min)

        A_grid = np.linspace(A_min, A_max, backbone_n)
        # k = 3/(16*np.pi**2)  # if your x-axis is in Hz
        k = 3/4               # if your x-axis is in rad/s
        inside = f0**2 + k * beta * A_grid**2
        mask = inside > 0.0
        f_grid = np.sqrt(inside[mask])
        A_plot = A_grid[mask]

        default_style = {"lw": 2, "label": "Backbone"}
        default_style.update(backbone_kwargs or {})
        (h_bb,) = ax.plot(f_grid, A_plot, **default_style)
        legend_handles.append(h_bb)

    # --- overlay measurement from get_data()
    if overlay_get_data:
        try:
            f_meas, x_meas = get_data()
            # Clean measurement
            f_meas = np.asarray(f_meas).ravel()
            x_meas = np.asarray(x_meas).ravel()
            m = np.isfinite(f_meas) & np.isfinite(x_meas)
            f_meas, x_meas = f_meas[m], x_meas[m]
            order = np.argsort(f_meas)
            f_meas, x_meas = f_meas[order], x_meas[order]

            # Build a simulation 'envelope' to compare against (use stable points if available)
            f_sim = np.asarray(freq_vals).ravel()
            x_sim = np.asarray(x_max).ravel()
            stab = np.asarray(stable).ravel().astype(bool) if stable is not None else None
            f_env, y_env = _envelope_by_bins(f_sim, x_sim, mask=stab, nbins=env_bins, agg=env_agg)

            # Decide alignment
            if align_measurement and f_env.size > 5:
                s, t, c = _align_measurement_to_sim(f_meas, x_meas, f_env, y_env, power=2.0)
            else:
                s, t, c = 1.0, 0.0, 1.0

            f_meas_mapped = s * f_meas + t
            x_meas_scaled = c * x_meas

            # 1) Raw aligned curve
            (h_meas_raw,) = ax.plot(
                f_meas_mapped,
                x_meas_scaled,
                lw=1.8,
                alpha=0.95,
                label="Measurement (aligned)",
            )
            legend_handles.append(h_meas_raw)

            # 2) Optional: markers at sim envelope frequencies (direct comparison)
            if show_interpolated and f_env.size > 0:
                # Only where we overlap
                overlap = (f_env >= f_meas_mapped.min()) & (f_env <= f_meas_mapped.max())
                if np.any(overlap):
                    x_on_env = np.interp(f_env[overlap], f_meas_mapped, x_meas_scaled)
                    (h_interp,) = ax.plot(
                        f_env[overlap],
                        x_on_env,
                        ls="none",
                        marker="x",
                        ms=4,
                        alpha=0.9,
                        label="Measurement (on sim grid)",
                    )
                    legend_handles.append(h_interp)

                # Optional: show the sim envelope itself for reference (thin line)
                (h_env,) = ax.plot(
                    f_env, y_env, lw=1.0, ls="--", alpha=0.6, label="Sim envelope"
                )
                legend_handles.append(h_env)

        except Exception as e:
            print(f"Could not overlay get_data(): {e}")

    # --- optional bifurcation overlays
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
        ax.legend(handles=legend_handles, title="Legend")

    ax.set_xlabel("Driving frequency")
    ax.set_ylabel("Steady-state displacement amplitude")
    ax.set_title(title)
    ax.grid(const.PLOT_GRID)
    plt.tight_layout()
    plt.savefig("branch_exploration.png", dpi=300)
    plt.show()