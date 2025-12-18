import h5py
import numpy as np

HDF5_FILENAME = "batch_2025-12-18_10:21:26_0.hdf5"
SIMULATION_ID = "simulation_000"

with h5py.File(HDF5_FILENAME, "r") as h5file:
    simulations_group = h5file["simulations"]
    sim_group = simulations_group[SIMULATION_ID]

    import matplotlib.pyplot as plt

    # Load displacement tensor: (n_freq, n_amp, n_init_vel, n_init_disp)
    unsweeped_total = sim_group["unsweeped_total"][...]
    n_freq, n_amp, n_init_vel, n_init_disp = unsweeped_total.shape

    # Try to get frequency and amplitude axes; fallback to indices if missing
    def get_axis(keys):
        for k in keys:
            if k in sim_group:
                return np.asarray(sim_group[k][...]).squeeze()
        return None

    freqs = get_axis(["frequencies", "freqs", "frequency", "f", "omega"])
    if freqs is None or freqs.shape[0] != n_freq:
        freqs = np.arange(n_freq)

    amps = get_axis(["amplitudes", "amps", "drive_amplitudes", "drive_amp", "A"])
    if amps is None or amps.shape[0] != n_amp:
        amps = np.arange(n_amp)

    # Plot: each amplitude in a different color, all init vel/disp points
    cmap = plt.cm.get_cmap("tab10", n_amp)
    plt.figure(figsize=(9, 6))

    # Precompute x repeated per frequency for all init conditions
    x = np.repeat(freqs, n_init_vel * n_init_disp)

    for a in range(n_amp):
        disp_a = unsweeped_total[:, a, :, :]            # (n_freq, n_init_vel, n_init_disp)
        y = disp_a.reshape(-1)                          # flatten all init vel/disp for each freq
        plt.scatter(x, y, s=6, alpha=0.5, color=cmap(a % cmap.N), label=f"amp={amps[a]}")

    plt.xlabel("Frequency")
    plt.ylabel("Displacement")
    plt.title(f"Displacement vs Frequency (all init vel/disp) — {SIMULATION_ID}")
    plt.legend(title="Amplitude", fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.show()
    