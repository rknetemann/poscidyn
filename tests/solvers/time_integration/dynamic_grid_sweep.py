import numpy as np
import poscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _extract_gamma_diagonal(gamma: np.ndarray):
    arr = np.asarray(gamma)
    if arr.ndim < 4:
        return arr.ravel().tolist()[: arr.size]

    diag = []
    max_modes = min(2, arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
    for i in range(max_modes):
        diag.append(float(arr[i, i, i, i]))
    return diag


def _format_param_text(Q: np.ndarray, omega_0: np.ndarray, gamma: np.ndarray, modal_forces: np.ndarray) -> str:
    q_vals = np.asarray(Q).ravel()
    omega_vals = np.asarray(omega_0).ravel()
    gamma_diag = _extract_gamma_diagonal(gamma)

    parts = []
    if q_vals.size:
        parts.append(f"Q=[{', '.join(f'{val:.2f}' for val in q_vals[:2])}]")
    if omega_vals.size:
        parts.append(f"omega0=[{', '.join(f'{val:.2f}' for val in omega_vals[:2])}]")
    if gamma_diag:
        parts.append(f"gamma=[{', '.join(f'{val:.2e}' for val in gamma_diag)}]")
    if modal_forces.size:
        parts.append(f"|F|=[{', '.join(f'{val:.2f}' for val in modal_forces[:2])}]")
    return "\n".join(parts)


def plot_sweep(ax, drive_freqs, drive_amps, sweeped_solutions, param_text: str) -> None:
    forward = sweeped_solutions.get("forward")
    backward = sweeped_solutions.get("backward")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions to plot.")

    drive_freqs = np.asarray(drive_freqs)
    drive_amps = np.asarray(drive_amps)
    colors = plt.cm.viridis(np.linspace(0, 1, drive_amps.size))

    for idx, (amp, color) in enumerate(zip(drive_amps, colors)):
        if forward is not None:
            ax.plot(
                drive_freqs,
                forward[:, idx],
                color=color,
                linestyle="-",
                linewidth=1.3,
            )
        if backward is not None:
            ax.plot(
                drive_freqs,
                backward[:, idx],
                color=color,
                linestyle="--",
                linewidth=1.1,
            )

    ax.set_title(f"Frequency sweep")
    ax.set_xlabel("Drive frequency")
    ax.set_ylabel("Max displacement")
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

    amp_handles = [
        Line2D([0], [0], color=color, lw=1.8) for color in colors
    ]
    amp_labels = [f"|F|={amp:.3f}" for amp in drive_amps]
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

    if forward is not None and backward is not None:
        style_handles = [
            Line2D([0], [0], color="k", linestyle="-", lw=1.5),
            Line2D([0], [0], color="k", linestyle="--", lw=1.5),
        ]
        style_labels = ["Forward sweep", "Backward sweep"]
        legend2 = ax.legend(
            style_handles,
            style_labels,
            loc="lower right",
            fontsize=7,
            frameon=False,
        )
        ax.add_artist(legend1)
    else:
        ax.add_artist(legend1)


def _merge_intervals(intervals, eps=1e-9):
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda pair: pair[0])
    merged = [list(sorted_intervals[0])]
    for start, end in sorted_intervals[1:]:
        if start - merged[-1][1] <= eps:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(start, end) for start, end in merged if end - start > eps]


def build_drive_frequency_grid(omega_0, fwhm, n_points=300, dense_multiplier=4.0, n_fwhm=4.0, min_freq=0.1):
    omega_0 = np.asarray(omega_0, dtype=float)
    fwhm = np.asarray(fwhm, dtype=float)
    if omega_0.size == 0:
        raise ValueError("omega_0 must contain at least one resonant frequency.")

    freq_min = max(min_freq, float(np.min(omega_0 - n_fwhm * fwhm)))
    freq_max = float(np.max(omega_0 + n_fwhm * fwhm))
    if freq_max <= freq_min:
        freq_max = freq_min + 1.0

    dense_multiplier = max(1.0, float(dense_multiplier))

    raw_windows = []
    for center, width in zip(omega_0, fwhm):
        start = max(freq_min, center - n_fwhm * width)
        end = min(freq_max, center + n_fwhm * width)
        if end > start:
            raw_windows.append((start, end))

    dense_segments = _merge_intervals(raw_windows)

    breakpoints = {freq_min, freq_max}
    for start, end in dense_segments:
        breakpoints.add(start)
        breakpoints.add(end)

    breakpoints = np.array(sorted(breakpoints))
    gaps = np.diff(breakpoints) > 1e-9
    if not np.any(gaps):
        return np.linspace(freq_min, freq_max, n_points)

    segment_starts = breakpoints[:-1][gaps]
    segment_ends = breakpoints[1:][gaps]
    segment_lengths = segment_ends - segment_starts

    def _is_dense(midpoint: float) -> bool:
        for start, end in dense_segments:
            if start - 1e-9 <= midpoint <= end + 1e-9:
                return True
        return False

    segment_weights = np.array(
        [
            dense_multiplier if _is_dense(0.5 * (lo + hi)) else 1.0
            for lo, hi in zip(segment_starts, segment_ends)
        ]
    )

    weighted_lengths = segment_lengths * segment_weights
    cdf = np.concatenate(([0.0], np.cumsum(weighted_lengths)))
    total_weighted = cdf[-1]
    if total_weighted <= 0:
        return np.linspace(freq_min, freq_max, n_points)

    targets = np.linspace(0.0, total_weighted, n_points)
    segment_indices = np.searchsorted(cdf[1:], targets, side="right")
    segment_indices = np.clip(segment_indices, 0, segment_lengths.size - 1)
    seg_offsets = targets - cdf[segment_indices]

    with np.errstate(divide="ignore", invalid="ignore"):
        local_param = np.divide(
            seg_offsets,
            weighted_lengths[segment_indices],
            out=np.zeros_like(seg_offsets),
            where=weighted_lengths[segment_indices] > 0,
        )

    drive_freqs = segment_starts[segment_indices] + local_param * segment_lengths[segment_indices]
    drive_freqs[0] = freq_min
    drive_freqs[-1] = freq_max
    return drive_freqs


# 1 mode:
# Q, omega_0, alpha, gamma = np.array([80.0]), np.array([1.0]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
# gamma[0,0,0,0] = 0.0123

# 2 modes:
Q, omega_0, alpha, gamma = np.array([30.0, 30.0]), np.array([2.00, 5.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
gamma[0,0,0,0] = 2.50e-3 * 0.001
gamma[1,1,1,1] = 5.00e-3 * 0

FWHM = omega_0 / Q
N_FWHM = 4.0
TARGET_FREQ_POINTS = 100
DENSE_MULTIPLIER = 1.0
MIN_DRIVE_FREQ = 0.1

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
DRIVING_FREQUENCY = build_drive_frequency_grid(
    omega_0=omega_0,
    fwhm=FWHM,
    n_points=TARGET_FREQ_POINTS,
    dense_multiplier=DENSE_MULTIPLIER,
    n_fwhm=N_FWHM,
    min_freq=MIN_DRIVE_FREQ,
)
print(f"Generated {DRIVING_FREQUENCY.size} drive frequencies for the sweep.")
print(DRIVING_FREQUENCY)
DRIVING_AMPLITUDE = np.linspace(10.0, 100.0, 10)
EXCITOR = poscidyn.OneToneExcitation(drive_frequencies=DRIVING_FREQUENCY, drive_amplitudes=DRIVING_AMPLITUDE, modal_forces=np.array([1.0, 0.5]))
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*3, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
SWEEPER = poscidyn.NearestNeighbourSweep(sweep_direction=[poscidyn.Forward(), poscidyn.Backward()])
PRECISION = poscidyn.Precision.SINGLE

start_time = time.time()

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL,
    sweeper=SWEEPER,
    excitor=EXCITOR,
    solver = SOLVER,
    precision = PRECISION,
) #n_freq, n_amp, n_init_disp, n_init_vel

end_time = time.time()
print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")

fig, ax = plt.subplots(figsize=(10, 6))
plot_sweep(
    ax=ax,
    drive_freqs=EXCITOR.drive_frequencies,
    drive_amps=EXCITOR.drive_amplitudes,
    sweeped_solutions=frequency_sweep.sweeped_periodic_solutions,
    param_text=_format_param_text(Q, omega_0, gamma, EXCITOR.modal_forces),
)
plt.tight_layout()
plt.show()
