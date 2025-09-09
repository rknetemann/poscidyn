import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

FILENAME = Path("/home/raymo/Downloads/batch_1.hdf5")

# ---- noise settings ----
SEED = 42

rng = np.random.default_rng(SEED)

# ---- load data ----
with h5py.File(FILENAME, "r") as f:
    freqs = np.array(f["driving_frequencies"][:], dtype=float)
    amps  = np.array(f["driving_amplitudes"][:], dtype=float)
    n_f, n_a = freqs.shape[0], amps.shape[0]

    total_disp_amp_low_Q = np.array(f["simulations/simulation_775"][:], dtype=float).reshape(n_f, n_a)
    total_disp_amp_high_Q = np.array(f["simulations/simulation_2475"][:], dtype=float).reshape(n_f, n_a)

# plot all force columns for both low and high Q, normalized by the response at the highest force
plt.figure(figsize=(7, 4.5))

# find the index of the largest driving amplitude
max_force_idx = np.argmax(amps)

# compute normalization factors from the max‐force column
norm_low  = total_disp_amp_low_Q[:,  max_force_idx].max()
norm_high = total_disp_amp_high_Q[:, max_force_idx].max()

for idx in range(n_a):
    # low Q (normalized)
    y_low = total_disp_amp_low_Q[:, idx] / norm_low
    plt.plot(freqs, y_low,
             lw=1.5, alpha=0.7,
             label=f"Norm |x| (Low Q), force {idx}")

for idx in range(n_a):
    # high Q (normalized)
    y_high = total_disp_amp_high_Q[:, idx] / norm_high
    plt.plot(freqs, y_high,
             lw=1.5, alpha=0.7, linestyle='--',
             label=f"Norm |x| (High Q), force {idx}")
    
plt.xlabel("ω [rad/s]")
plt.ylabel("|x| (arb. units)")
plt.title("Frequency Response: Clean vs Noisy Data")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
