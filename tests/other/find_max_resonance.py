import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

FILENAME = Path("/home/raymo/Downloads/batch_1.hdf5")
FORCE_IDX = 9

# ---- noise settings (same semantics as your script) ----
SEED = 42
REL_NOISE = 0.05
ADD_NOISE_FRAC = 0.02
CLIP_NONNEG = True

rng = np.random.default_rng(SEED)

def add_noise(y, rel_sigma=REL_NOISE, add_sigma_frac=ADD_NOISE_FRAC, rng=None, clip_nonneg=True):
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, float)
    y_max = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    add_sigma = add_sigma_frac * y_max
    y_noisy = y * (1.0 + rng.normal(0.0, rel_sigma, size=y.shape)) + rng.normal(0.0, add_sigma, size=y.shape)
    if clip_nonneg:
        y_noisy = np.maximum(y_noisy, 0.0)
    return y_noisy

# ---- load data ----
with h5py.File(FILENAME, "r") as f:
    freqs = np.array(f["driving_frequencies"][:], dtype=float)
    amps  = np.array(f["driving_amplitudes"][:], dtype=float)
    n_f, n_a = freqs.shape[0], amps.shape[0]

    total_disp_amp = np.array(f["simulations/simulation_2499"][:], dtype=float).reshape(n_f, n_a)

# pick one force column (frequency sweep at FORCE_IDX)
y_clean = total_disp_amp[:, FORCE_IDX]

# optional: add measurement noise (use y_target to choose which curve to analyze)
y_noisy = add_noise(y_clean, rng=rng, rel_sigma=REL_NOISE, add_sigma_frac=ADD_NOISE_FRAC, clip_nonneg=CLIP_NONNEG)
y_target = y_noisy           # change to y_clean if you want the max of the clean data

# ---- find maximum amplitude ----
if np.all(~np.isfinite(y_target)):
    raise ValueError("All values are non-finite; cannot determine a maximum.")

idx_max = np.nanargmax(y_target)        # ignores NaNs
f_star  = float(freqs[idx_max])
y_star  = float(y_target[idx_max])

print(f"Max displacement amplitude: {y_star:.6g} at ω = {f_star:.6g} rad/s (index {idx_max})")

# ---- plot ----
plt.figure(figsize=(7, 4.5))
plt.plot(freqs, y_clean, lw=1.5, alpha=0.9, label="Clean |x|")
plt.scatter(freqs, y_noisy, s=12, alpha=0.6, label="Noisy |x|")

# highlight the maximum point from y_target
plt.scatter([f_star], [y_star], s=100, marker="o", edgecolors="k", linewidths=1.0, zorder=5, label="Max |x|")
# plt.annotate(f"max = {y_star:.3g}\nω = {f_star:.3g}",
#              xy=(f_star, y_star), xytext=(10, 12), textcoords="offset points",
#              ha="left", va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="w", alpha=0.8))

plt.xlabel("ω [rad/s]")
plt.ylabel("|x| (arb. units)")
plt.title("Maximum Displacement Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
