import numpy as np
import h5py
import matplotlib.pyplot as plt

FILENAME = "/home/raymo/Downloads/batch_1.hdf5"
FORCE_IDX = 9

# ---- noise settings ----
SEED = 42                 # for reproducibility
REL_NOISE = 0.05          # multiplicative std as fraction of amplitude (e.g., 5%)
ADD_NOISE_FRAC = 0.02     # additive std as fraction of max amplitude (e.g., 2%)
CLIP_NONNEG = True        # amplitudes must be ≥ 0 for log-domain fitting

rng = np.random.default_rng(SEED)

def add_noise(y, rel_sigma=REL_NOISE, add_sigma_frac=ADD_NOISE_FRAC, rng=None, clip_nonneg=True):
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, float)
    y_max = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    add_sigma = add_sigma_frac * y_max

    mult = 1.0 + rng.normal(0.0, rel_sigma, size=y.shape)        # multiplicative
    add  = rng.normal(0.0, add_sigma, size=y.shape)               # additive
    y_noisy = y * mult + add
    if clip_nonneg:
        y_noisy = np.maximum(y_noisy, 0.0)
    return y_noisy

def resonance_model_amp(omega, A0, w0, zeta):
    """
    Amplitude-only model |x(ω)| for a 2nd-order SDOF under sinusoidal force.
    A0 = F/k (vertical scale), w0 = sqrt(k/m), zeta = c/(2*sqrt(k*m))
    """
    omega = np.asarray(omega, dtype=float)
    r = omega / w0
    denom = np.sqrt((1.0 - r**2)**2 + (2.0*zeta*r)**2)
    return A0 / denom

def fit_resonance(omega, y,
                  w0_bounds=(1e-6, np.inf),
                  zeta_bounds=(1e-6, 2.0),
                  w0_grid=None,
                  zeta_grid=None,
                  local_refine_iters=3):
    """
    Grid-search (w0, zeta) + closed-form A0 to fit |x(ω)| data.

    Parameters
    ----------
    omega : array-like
        Driving angular frequencies [rad/s].
    y : array-like
        Measured steady-state amplitudes |x|.
    w0_bounds : (low, high)
        Bounds for natural frequency (rad/s).
    zeta_bounds : (low, high)
        Bounds for damping ratio (dimensionless).
    w0_grid : array-like or None
        Optional grid for w0. If None, it is auto-generated from data span.
    zeta_grid : array-like or None
        Optional grid for zeta. If None, linear grid in given bounds.
    local_refine_iters : int
        Simple coordinate search steps around the best grid point.

    Returns
    -------
    params : dict
        {'A0': ..., 'w0': ..., 'zeta': ..., 'sse': ...}
    y_fit : np.ndarray
        Fitted amplitudes at the provided omega.
    """
    omega = np.asarray(omega, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    assert omega.shape == y.shape, "omega and y must have same shape"

    # Build default grids if not provided
    if w0_grid is None:
        pos = omega[omega > 0]
        w_med = np.median(pos) if pos.size else 1.0
        w_max = np.max(omega) if np.max(omega) > 0 else 1.0
        w_low = max(w0_bounds[0], 0.25 * w_med)
        w_high = min(w0_bounds[1], max(2.0 * w_max, 4.0 * w_med))
        w0_grid = np.geomspace(w_low, w_high, 120)

    if zeta_grid is None:
        z_lo = max(1e-6, zeta_bounds[0])
        z_hi = min(2.0, zeta_bounds[1])
        zeta_grid = np.linspace(z_lo, z_hi, 120)

    best = dict(err=np.inf, A0=None, w0=None, zeta=None)

    # Grid search over (w0, zeta) with closed-form A0
    for w0 in w0_grid:
        r = omega / w0
        base = 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * zeta_grid[:, None] * r)**2)  # (Z, N)
        # A0* per row minimizes || y - A0*base ||_2
        dot = base @ y
        nrm2 = np.sum(base**2, axis=1)
        A0_rows = dot / np.maximum(nrm2, 1e-12)
        y_pred_rows = A0_rows[:, None] * base
        errs = np.sum((y_pred_rows - y)**2, axis=1)
        j = np.argmin(errs)
        if errs[j] < best["err"]:
            best.update(err=float(errs[j]), A0=float(A0_rows[j]),
                        w0=float(w0), zeta=float(zeta_grid[j]))

    # Lightweight local refinement (coordinate search)
    for _ in range(int(local_refine_iters)):
        improved = False
        # refine w0
        for w0 in best["w0"] * np.geomspace(0.9, 1.1, 21):
            r = omega / w0
            base = 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * best["zeta"] * r)**2)
            A0 = (base @ y) / max(np.sum(base**2), 1e-12)
            err = np.sum((A0 * base - y)**2)
            if err < best["err"]:
                best.update(err=float(err), A0=float(A0), w0=float(w0))
                improved = True
        # refine zeta
        for z in np.clip(best["zeta"] * np.geomspace(0.8, 1.25, 21), 1e-6, 2.0):
            r = omega / best["w0"]
            base = 1.0 / np.sqrt((1.0 - r**2)**2 + (2.0 * z * r)**2)
            A0 = (base @ y) / max(np.sum(base**2), 1e-12)
            err = np.sum((A0 * base - y)**2)
            if err < best["err"]:
                best.update(err=float(err), A0=float(A0), zeta=float(z))
                improved = True
        if not improved:
            break

    y_fit = resonance_model_amp(omega, best["A0"], best["w0"], best["zeta"])
    return {"A0": best["A0"], "w0": best["w0"], "zeta": best["zeta"], "sse": best["err"]}, y_fit


with h5py.File(FILENAME, "r") as hdf5:
    freqs = np.array(hdf5["driving_frequencies"][:], dtype=np.float32) * 1
    amps  = np.array(hdf5["driving_amplitudes"][:], dtype=np.float32)
    n_f, n_a = freqs.shape[0], amps.shape[0]

    total_disp_amp = np.array(hdf5["simulations/simulation_2499"][:], dtype=np.float32)
    total_disp_amp = total_disp_amp.reshape((n_f, n_a))

    y_clean = total_disp_amp[:, FORCE_IDX]
    y_noisy = add_noise(y_clean, rel_sigma=REL_NOISE, add_sigma_frac=ADD_NOISE_FRAC, rng=rng, clip_nonneg=CLIP_NONNEG)

    # Fit on the noisy data; if you want tail-prioritized fitting, pass the options:
    params, y_fit = fit_resonance(
        freqs, y_noisy,
        # tail-focused options (uncomment if your fit_resonance supports them)
        # weight_mode="tail",
        # exclude_peak_frac=0.8,
        # dist_power=2.0,
        # use_log_amp=True
    )
    print("Fitted parameters (noisy data):", params)

    # Plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(freqs, y_clean, lw=1.5, alpha=0.9, label="Clean |x|")
    plt.scatter(freqs, y_noisy, s=10, alpha=0.6, label="Noisy |x|")
    plt.plot(freqs, y_fit, lw=2.0, label="Fitted model (noisy)")
    plt.xlabel("ω [rad/s]"); plt.ylabel("|x| (arb. units)")
    plt.title("Linear Resonance Peak Fit with Noise")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()