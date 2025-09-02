import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

FILENAME = Path("/home/raymo/Downloads/batch_1.hdf5")
FORCE_IDX = 9

# ---- noise settings ----
SEED = 42
REL_NOISE = 0.05       # multiplicative std (fraction of amplitude)
ADD_NOISE_FRAC = 0.02  # additive std (fraction of max amplitude)
CLIP_NONNEG = True

rng = np.random.default_rng(SEED)

def add_noise(y, rel_sigma=REL_NOISE, add_sigma_frac=ADD_NOISE_FRAC, rng=None, clip_nonneg=True):
    if rng is None:
        rng = np.random.default_rng()
    y = np.asarray(y, float)
    y_max = np.nanmax(np.abs(y)) if np.any(np.isfinite(y)) else 1.0
    add_sigma = add_sigma_frac * y_max
    y_noisy = y * (1.0 + rng.normal(0.0, rel_sigma, size=y.shape)) \
                + rng.normal(0.0, add_sigma, size=y.shape)
    if clip_nonneg:
        y_noisy = np.maximum(y_noisy, 0.0)
    return y_noisy

# ---- load data ----
with h5py.File(FILENAME, "r") as f:
    freqs = np.array(f["driving_frequencies"][:], dtype=float)
    amps  = np.array(f["driving_amplitudes"][:], dtype=float)
    n_f, n_a = freqs.shape[0], amps.shape[0]

    total_disp_amp = np.array(f["simulations/simulation_2499"][:], dtype=float).reshape(n_f, n_a)

# pick one force column
y_clean = total_disp_amp[:, FORCE_IDX]
y_noisy = add_noise(y_clean, rng=rng, rel_sigma=REL_NOISE,
                    add_sigma_frac=ADD_NOISE_FRAC, clip_nonneg=CLIP_NONNEG)

# ---- plot clean + noisy ----
plt.figure(figsize=(7, 4.5))
plt.plot(freqs, y_clean, lw=1.8, alpha=0.9, label="Clean |x|")
plt.scatter(freqs, y_noisy, s=10, alpha=0.6, label="Noisy |x|")

plt.xlabel("ω [rad/s]")
plt.ylabel("|x| (arb. units)")
plt.title("Frequency Response: Clean vs Noisy Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
