import numpy as np
import poscidyn
import time
import h5py  # <-- NEW

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================
# NEW: load original A from file
# =========================
A_NPZ_PATH = "/home/raymo/Projects/nlsid/worst_gamma_case_A_only.npz"  # <-- set your npz path
HDF5_DATASET_PATH = "/home/raymo/Projects/nlsid/datasets/2026012301_string_2DOF/2026012301_string_2DOF_augmented_split.hdf5"  # <-- set your dataset file
SPLIT = "test"          # <-- set train/val/test
HDF5_INDEX = None       # <-- optional: override; if None we'll use worst_index from the npz

# If you want to plot the dataset A as-is (already normalized), keep True.
# If you want to undo the normalization used in plot_sweep() for the simulation (divide by x_ref),
# set this False and we'll normalize the dataset A the same way before plotting.
PLOT_DATASET_A_AS_STORED = True


def F_max(eta, omega_0, Q, gamma):
    return np.sqrt(
        4 * omega_0**6
        / (3 * gamma * Q**2)
        * (eta + 1 / (2 * Q**2))
        * (1 + eta + 1 / (4 * Q**2))
    )


# 1 mode:
Q, omega_0, alpha, gamma = np.array([100.0]), np.array([1.00]), np.zeros((1, 1, 1)), np.zeros((1, 1, 1, 1))
gamma[0, 0, 0, 0] = 2.55
modal_forces = np.array([1.0])

# 2 modes:
Q, omega_0, alpha, gamma = (
    np.array([6.357357, 36.742565]),
    np.array([0.9946662, 2.1275177]),
    np.zeros((2, 2, 2)),
    np.zeros((2, 2, 2, 2)),
)
gamma[0, 0, 0, 0] = 1.9451260e01
gamma[0, 0, 1, 1] = 1.3927963e02
gamma[1, 1, 1, 1] = 2.6890811e02
gamma[1, 0, 0, 1] = 1.4860869e02

F_max_value = F_max(0.20, omega_0[0], Q[0], gamma[0, 0, 0, 0])
print(f"Calculated F_max: {F_max_value:.4f}")


# You can keep these as you had them:
driving_frequency = np.linspace(0.6322645545005798, 2.2314629554748535, 400)
driving_amplitude = np.linspace(0.1, 1.0, 10) * 1
modal_forces = np.array([0.01101661, 0.00740971])

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(
    max_steps=4096 * 5, n_time_steps=50, verbose=True, throw=False, rtol=1e-4, atol=1e-7
)
SWEEPER = poscidyn.NearestNeighbourSweep(sweep_direction=[poscidyn.Forward(), poscidyn.Backward()])
PRECISION = poscidyn.Precision.SINGLE


def _extract_gamma_entries(gamma: np.ndarray, max_entries: int = 3):
    arr = np.asarray(gamma)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _extract_alpha_entries(alpha: np.ndarray, max_entries: int = 3):
    arr = np.asarray(alpha)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _format_param_text(Q: np.ndarray, omega_0: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, modal_forces: np.ndarray) -> str:
    q_vals = np.asarray(Q).ravel()
    omega_vals = np.asarray(omega_0).ravel()
    alpha_entries, alpha_total = _extract_alpha_entries(alpha)
    gamma_entries, gamma_total = _extract_gamma_entries(gamma)

    parts = []
    if q_vals.size:
        parts.append(f"Q=[{', '.join(f'{val:.2f}' for val in q_vals[:2])}]")
    if omega_vals.size:
        parts.append(f"omega0=[{', '.join(f'{val:.2f}' for val in omega_vals[:2])}]")
    if alpha_entries:
        formatted_alpha = ", ".join(f"{idx}={val:.2e}" for idx, val in alpha_entries)
        if alpha_total > len(alpha_entries):
            formatted_alpha += ", ..."
        parts.append(f"alpha={formatted_alpha}")
    if gamma_entries:
        formatted_gamma = ", ".join(f"{idx}={val:.2e}" for idx, val in gamma_entries)
        if gamma_total > len(gamma_entries):
            formatted_gamma += ", ..."
        parts.append(f"gamma={formatted_gamma}")
    if modal_forces.size:
        parts.append(f"modal_forces=[{', '.join(f'{val:.2f}' for val in modal_forces[:2])}]")
    return "\n".join(parts)


# =========================
# NEW: load the stored dataset curve + its omega range
# =========================
def load_dataset_curve():
    idx = HDF5_INDEX
    if idx is None:
        # Try to pull index from the npz you saved earlier
        d = np.load(A_NPZ_PATH)
        if "worst_index" in d:
            idx = int(d["worst_index"])
        else:
            raise ValueError("HDF5_INDEX is None and NPZ file does not contain 'worst_index'.")

    with h5py.File(HDF5_DATASET_PATH, "r") as f:
        g = f[SPLIT]
        A_ds = np.asarray(g["A"][idx, :], dtype=np.float64) * (50.0 / 2.347)* 0.9939844
        fmin = float(np.asarray(g["f_omega_min"][idx]))
        fmax = float(np.asarray(g["f_omega_max"][idx]))
        omega_axis = np.linspace(fmin, fmax, A_ds.shape[0], dtype=np.float64)

    return idx, omega_axis, A_ds, fmin, fmax


# =========================
# MODIFIED: plot_sweep now also plots dataset A
# =========================
def plot_sweep(ax, drive_freqs, drive_amps, sweeped_solutions, param_text: str) -> None:
    forward = sweeped_solutions.get("forward")
    backward = sweeped_solutions.get("backward")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions to plot.")

    # --- reference normalization for the simulation (your existing logic)
    ref_idx = np.argmin(np.abs(drive_freqs - 0.9 * omega_0[0]))
    omega_ref = drive_freqs[ref_idx]

    x_ref_forward = forward[ref_idx, :]
    x_ref_backward = backward[ref_idx, :]

    print(f"Reference frequency: {omega_ref}")
    print(f"Reference displacement forward: {x_ref_forward}, backward: {x_ref_backward}")

    forward = forward / x_ref_forward
    backward = backward / x_ref_backward

    drive_freqs_norm = np.asarray(drive_freqs, dtype=np.float64) / omega_ref
    drive_amps = np.asarray(drive_amps)
    colors = plt.cm.viridis(np.linspace(0, 1, drive_amps.size))

    # --- simulation curves
    for idx, (amp, color) in enumerate(zip(drive_amps, colors)):
        if forward is not None:
            ax.plot(drive_freqs_norm, forward[:, idx], color=color, linestyle="-", linewidth=1.3)
        if backward is not None:
            ax.plot(drive_freqs_norm, backward[:, idx], color=color, linestyle="--", linewidth=1.1)

    # --- NEW: overlay dataset A (as saved)
    ds_idx, omega_ds, A_ds, fmin_ds, fmax_ds = load_dataset_curve()

    # dataset frequency axis normalized the same way as simulation:
    omega_ds_norm = omega_ds / omega_ref

    if PLOT_DATASET_A_AS_STORED:
        A_ds_plot = A_ds
    else:
        # normalize dataset A with the same x_ref strategy:
        # (use the dataset’s own omega axis to find nearest 0.9*omega0[0])
        target = 0.9 * float(omega_0[0])
        ref_idx_ds = int(np.argmin(np.abs(omega_ds - target)))
        x_ref_ds = float(A_ds[ref_idx_ds])
        A_ds_plot = A_ds / x_ref_ds

    ax.plot(
        omega_ds_norm,
        A_ds_plot,
        color="black",
        linewidth=2.2,
        alpha=0.9,
        label=f"Dataset A (idx={ds_idx})",
        zorder=6,
    )

    # also mark the dataset reference point (0.9*omega0[0]) on the dataset curve
    target = 0.9 * float(omega_0[0])
    ref_idx_ds = int(np.argmin(np.abs(omega_ds - target)))
    ax.scatter(
        [omega_ds_norm[ref_idx_ds]],
        [A_ds_plot[ref_idx_ds]],
        color="red",
        s=55,
        zorder=7,
        label="0.9·ω₀[0] ref (dataset)",
    )

    ax.set_title("Frequency sweep (simulation) + stored dataset A overlay")
    ax.set_xlabel("Drive frequency (normalized by ω_ref)")
    ax.set_ylabel("Max displacement (simulation normalized by x_ref)")
    ax.grid(alpha=0.25)

    if param_text:
        ax.text(
            0.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
        )

    # Legends
    amp_handles = [Line2D([0], [0], color=color, lw=1.8) for color in colors]
    amp_labels = [f"F={amp:.3f}" for amp in drive_amps]
    legend1 = ax.legend(
        amp_handles,
        amp_labels,
        title="Drive amplitude",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7,
        frameon=False,
        ncol=2,
    )

    style_handles = [
        Line2D([0], [0], color="k", linestyle="-", lw=1.5),
        Line2D([0], [0], color="k", linestyle="--", lw=1.5),
        Line2D([0], [0], color="black", linestyle="-", lw=2.2),
        Line2D([0], [0], color="red", marker="o", linestyle="None", markersize=6),
    ]
    style_labels = ["Forward sweep", "Backward sweep", "Dataset A", "Dataset ref point"]

    legend2 = ax.legend(
        style_handles,
        style_labels,
        loc="lower right",
        fontsize=7,
        frameon=False,
    )
    ax.add_artist(legend1)


start_time = time.time()

frequency_sweep = poscidyn.frequency_sweep(
    model=MODEL,
    sweeper=SWEEPER,
    excitor=EXCITOR,
    solver=SOLVER,
    precision=PRECISION,
    multistarter=MULTISTART,
)  # n_freq, n_amp, n_init_disp, n_init_vel

end_time = time.time()
print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")
print(
    f"Successful periodic solutions: {frequency_sweep.n_successful}/{frequency_sweep.n_total} "
    f"({frequency_sweep.success_rate:.1%})"
)

fig, ax = plt.subplots(figsize=(10, 6))
plot_sweep(
    ax=ax,
    drive_freqs=EXCITOR.drive_frequencies,
    drive_amps=EXCITOR.drive_amplitudes,
    sweeped_solutions=frequency_sweep.sweeped_periodic_solutions,
    param_text=_format_param_text(Q, omega_0, alpha, gamma, EXCITOR.modal_forces),
)
plt.tight_layout()
plt.show()
