import numpy as np
import poscidyn
import time
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def F_max(eta, omega_0, Q, b):
    return np.sqrt(
        4 * omega_0**6 / (3 * b * Q**2)
        * (eta + 1 / (2 * Q**2))
        * (1 + eta + 1 / (4 * Q**2))
    )


# ============================================================
# Model definition
# ============================================================

# 1 mode example:
# Q, omega_0, a, b = np.array([50.0]), np.array([1.00]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
# b[0,0,0,0] = 2.55
# modal_forces = np.array([1.0])

# 2 mode example:
Q = np.array([50.0, 80.0])
omega_0 = np.array([1.0, 1.5])
a = np.zeros((2, 2, 2))
b = np.zeros((2, 2, 2, 2))
b[0, 0, 0, 0] = 1.0
b[1, 1, 1, 1] = -0.5
# a[0,0,1] = 2.0 * 1 * 0.08
# a[1,0,0] = 1 * 0.08
modal_forces = np.array([1.0, 1.0])

# ------------------------------------------------------------
# Mode shape vector at the MEASUREMENT location r_m
# phi_i(r_m) for each mode i
#
# Example:
# - mode 1 contributes fully
# - mode 2 contributes with smaller weight
#
# Replace with your actual mode shape values at the probe point.
# ============================================================
phi_rm = np.array([1.0, 0.9], dtype=float)

F_max_value = F_max(0.20, omega_0[0], Q[0], b[0, 0, 0, 0])
print(f"Calculated F_max: {F_max_value:.4f}")

driving_frequency = np.linspace(0.8, 2.0, 400)
driving_amplitude = np.linspace(0.1, 1.0, 10) * F_max_value

MODEL = poscidyn.NonlinearOscillator(omega_0=omega_0, Q=Q,a=a, b=b)
print(MODEL)
EXCITATION = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(
    max_steps=4096 * 5,
    n_time_steps=50,
    verbose=True,
    throw=False,
    rtol=1e-4,
    atol=1e-7,
)
SWEEPER = poscidyn.NearestNeighbourSweep(
    sweep_direction=[poscidyn.Forward(), poscidyn.Backward()]
)
RESPONSE_MEASURE = poscidyn.Demodulation(multiples=(1,))
# RESPONSE_MEASURE = poscidyn.L2(mode_shape=phi_rm)
PRECISION = poscidyn.Precision.SINGLE


# ============================================================
# Helpers
# ============================================================

def _extract_b_entries(b: np.ndarray, max_entries: int = 3):
    arr = np.asarray(b)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _extract_a_entries(a: np.ndarray, max_entries: int = 3):
    arr = np.asarray(a)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _format_param_text(
    Q: np.ndarray,
    omega_0: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    modal_forces: np.ndarray,
    phi_rm: np.ndarray | None = None,
) -> str:
    q_vals = np.asarray(Q).ravel()
    omega_vals = np.asarray(omega_0).ravel()
    a_entries, a_total = _extract_a_entries(a)
    b_entries, b_total = _extract_b_entries(b)

    parts = []
    if q_vals.size:
        parts.append(f"Q=[{', '.join(f'{val:.2f}' for val in q_vals[:4])}]")
    if omega_vals.size:
        parts.append(f"omega0=[{', '.join(f'{val:.2f}' for val in omega_vals[:4])}]")
    if a_entries:
        formatted_a = ", ".join(f"{idx}={val:.2e}" for idx, val in a_entries)
        if a_total > len(a_entries):
            formatted_a += ", ..."
        parts.append(f"a={formatted_a}")
    if b_entries:
        formatted_b = ", ".join(f"{idx}={val:.2e}" for idx, val in b_entries)
        if b_total > len(b_entries):
            formatted_b += ", ..."
        parts.append(f"b={formatted_b}")
    if modal_forces.size:
        parts.append(f"modal_forces=[{', '.join(f'{val:.2f}' for val in modal_forces[:4])}]")
    if phi_rm is not None:
        phi_vals = np.asarray(phi_rm).ravel()
        parts.append(f"phi_rm=[{', '.join(f'{val:.3f}' for val in phi_vals[:4])}]")
    return "\n".join(parts)


def _to_4d_response(arr: np.ndarray) -> np.ndarray:
    """
    Normalize response to shape:
        (n_freq, n_amp, n_multiples, n_modes)
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[:, :, None, None]
    if arr.ndim == 3:
        return arr[:, :, None, :]
    if arr.ndim == 4:
        return arr
    raise ValueError(f"Unsupported response shape: {arr.shape}")


def _finite_limits(data: np.ndarray, pad: float = 0.05) -> tuple[float, float]:
    data = np.asarray(data)
    finite = np.isfinite(data)
    if not np.any(finite):
        return -1.0, 1.0

    y_min = float(np.min(data[finite]))
    y_max = float(np.max(data[finite]))
    if y_min == y_max:
        eps = 1e-6 if y_min == 0.0 else abs(y_min) * 0.05
        return y_min - eps, y_max + eps

    span = y_max - y_min
    return y_min - pad * span, y_max + pad * span


def _phasors_to_sweeped_solutions(phasors) -> dict:
    amplitudes = phasors.amplitudes
    phases = phasors.phases
    demod_freqs = phasors.demod_freqs

    if not isinstance(amplitudes, dict):
        raise TypeError("phasors.amplitudes must be a dict with forward/backward keys.")

    return {
        "forward": amplitudes.get("forward"),
        "backward": amplitudes.get("backward"),
        "forward_phase": None if phases is None else phases.get("forward"),
        "backward_phase": None if phases is None else phases.get("backward"),
        "forward_demod_freq": None if demod_freqs is None else demod_freqs.get("forward"),
        "backward_demod_freq": None if demod_freqs is None else demod_freqs.get("backward"),
        "forward_idx": None,
        "backward_idx": None,
    }


def append_total_response_mode(
    modal_sweeped_solutions: dict,
    total_sweeped_solutions: dict,
    mode_labels: list[str] | None = None,
) -> tuple[dict, list[str]]:
    forward = modal_sweeped_solutions.get("forward")
    backward = modal_sweeped_solutions.get("backward")
    forward_phase = modal_sweeped_solutions.get("forward_phase")
    backward_phase = modal_sweeped_solutions.get("backward_phase")

    forward_total = total_sweeped_solutions.get("forward")
    backward_total = total_sweeped_solutions.get("backward")
    forward_total_phase = total_sweeped_solutions.get("forward_phase")
    backward_total_phase = total_sweeped_solutions.get("backward_phase")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions found.")

    out = dict(modal_sweeped_solutions)

    if mode_labels is None:
        template = forward if forward is not None else backward
        n_modes = _to_4d_response(template).shape[-1]
        mode_labels = [f"mode {i}" for i in range(n_modes)]

    def _combine(
        modal_amplitude: np.ndarray,
        modal_phase: np.ndarray,
        total_amplitude: np.ndarray,
        total_phase: np.ndarray,
    ):
        modal_amp4 = _to_4d_response(modal_amplitude)
        modal_phase4 = _to_4d_response(modal_phase)
        total_amp4 = _to_4d_response(total_amplitude)
        total_phase4 = _to_4d_response(total_phase)

        if modal_amp4.shape[:3] != total_amp4.shape[:3]:
            raise ValueError(
                "Modal and total response shapes do not match in (freq, amp, multiples)."
            )

        amp_out = np.concatenate([modal_amp4, total_amp4], axis=-1)
        phase_out = np.concatenate([modal_phase4, total_phase4], axis=-1)
        return amp_out, phase_out

    if forward is not None:
        out["forward"], out["forward_phase"] = _combine(
            forward,
            forward_phase,
            forward_total,
            forward_total_phase,
        )

    if backward is not None:
        out["backward"], out["backward_phase"] = _combine(
            backward,
            backward_phase,
            backward_total,
            backward_total_phase,
        )

    mode_labels = list(mode_labels) + ["total"]
    return out, mode_labels


def plot_sweep_grid(
    drive_freqs,
    drive_amps,
    sweeped_solutions,
    param_text: str,
    multiples: np.ndarray | None = None,
    mode_labels: list[str] | None = None,
):
    forward = sweeped_solutions.get("forward")
    backward = sweeped_solutions.get("backward")
    forward_phase = sweeped_solutions.get("forward_phase")
    backward_phase = sweeped_solutions.get("backward_phase")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions to plot.")

    if forward_phase is None:
        forward_phase = np.full_like(forward if forward is not None else backward, np.nan)
    if backward_phase is None:
        backward_phase = np.full_like(backward if backward is not None else forward, np.nan)

    drive_freqs = np.asarray(drive_freqs)
    drive_amps = np.asarray(drive_amps)
    colors = plt.cm.viridis(np.linspace(0, 1, drive_amps.size))

    fwd4 = _to_4d_response(forward) if forward is not None else None
    bwd4 = _to_4d_response(backward) if backward is not None else None
    fwdp4 = _to_4d_response(forward_phase)
    bwdp4 = _to_4d_response(backward_phase)

    amp_stack = [arr for arr in (fwd4, bwd4) if arr is not None]
    amp_all = np.concatenate(amp_stack, axis=1)
    amp_ylim = _finite_limits(amp_all)

    template = fwd4 if fwd4 is not None else bwd4
    _, _, n_multiples, n_modes = template.shape

    if multiples is None:
        multiple_labels = [f"{i}" for i in range(n_multiples)]
    else:
        multiples = np.asarray(multiples).reshape(-1)
        if multiples.size == n_multiples:
            multiple_labels = [f"{m:g}" for m in multiples]
        else:
            multiple_labels = [f"{i}" for i in range(n_multiples)]

    if mode_labels is None:
        mode_labels = [f"mode {i}" for i in range(n_modes)]
    if len(mode_labels) != n_modes:
        raise ValueError(
            f"len(mode_labels) = {len(mode_labels)} but n_modes = {n_modes}"
        )

    n_rows = n_modes * n_multiples
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, max(4, 2.8 * n_rows)),
        sharex=True,
    )
    axes = np.atleast_2d(axes)

    for mode_idx in range(n_modes):
        for mult_idx in range(n_multiples):
            row_idx = mode_idx * n_multiples + mult_idx
            ax_amp = axes[row_idx, 0]
            ax_phase = axes[row_idx, 1]

            for amp_idx, (amp, color) in enumerate(zip(drive_amps, colors)):
                if fwd4 is not None:
                    ax_amp.plot(
                        drive_freqs,
                        fwd4[:, amp_idx, mult_idx, mode_idx],
                        color=color,
                        linestyle="-",
                        linewidth=1.1,
                    )
                    ax_phase.plot(
                        drive_freqs,
                        -fwdp4[:, amp_idx, mult_idx, mode_idx],
                        color=color,
                        linestyle="-",
                        linewidth=1.1,
                    )

                if bwd4 is not None:
                    ax_amp.plot(
                        drive_freqs,
                        bwd4[:, amp_idx, mult_idx, mode_idx],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                    )
                    ax_phase.plot(
                        drive_freqs,
                        -bwdp4[:, amp_idx, mult_idx, mode_idx],
                        color=color,
                        linestyle="--",
                        linewidth=1.0,
                    )

            ax_amp.set_ylabel(f"A ({mode_labels[mode_idx]}, mult {multiple_labels[mult_idx]})")
            ax_phase.set_ylabel(f"phi ({mode_labels[mode_idx]}, mult {multiple_labels[mult_idx]})")
            ax_amp.set_ylim(*amp_ylim)
            ax_amp.grid(alpha=0.25)
            ax_phase.grid(alpha=0.25)

    axes[-1, 0].set_xlabel("Drive frequency")
    axes[-1, 1].set_xlabel("Drive frequency")

    amp_handles = [Line2D([0], [0], color=color, lw=1.6) for color in colors]
    amp_labels = [f"F={amp:.3f}" for amp in drive_amps]

    style_handles = [
        Line2D([0], [0], color="k", linestyle="-", lw=1.4),
        Line2D([0], [0], color="k", linestyle="--", lw=1.4),
    ]
    style_labels = ["Forward sweep", "Backward sweep"]

    legend = axes[0, 0].legend(
        amp_handles + style_handles,
        amp_labels + style_labels,
        title="Drive amplitude / sweep",
        loc="best",
        fontsize=7,
        frameon=False,
        ncol=2,
    )
    axes[0, 0].add_artist(legend)

    fig.suptitle("Frequency sweep")
    if param_text:
        fig.text(
            0.01,
            0.99,
            param_text,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
        )

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ============================================================
# Run sweep
# ============================================================

start_time = time.time()

frequency_sweep = poscidyn.frequency_sweep(
    model=MODEL,
    sweeper=SWEEPER,
    excitation=EXCITATION,
    solver=SOLVER,
    response_measure=RESPONSE_MEASURE,
    precision=PRECISION,
    multistarter=MULTISTART,
)

end_time = time.time()
print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")
n_successful = frequency_sweep.stats["n_successful"]
n_total = frequency_sweep.stats["n_total"]
success_rate = frequency_sweep.stats["success_rate"]
print(
    f"Successful periodic solutions: {n_successful}/{n_total} "
    f"({success_rate:.1%})"
)

# ============================================================
# Reconstruct total measured displacement at r_m
# ============================================================

modal_sweeped_solutions = _phasors_to_sweeped_solutions(frequency_sweep.modal_coordinates)
total_sweeped_solutions = _phasors_to_sweeped_solutions(frequency_sweep.modal_superposition)

sweeped_with_total, mode_labels = append_total_response_mode(
    modal_sweeped_solutions=modal_sweeped_solutions,
    total_sweeped_solutions=total_sweeped_solutions,
)

# ============================================================
# Plot
# ============================================================

fig = plot_sweep_grid(
    drive_freqs=EXCITATION.drive_frequencies,
    drive_amps=EXCITATION.drive_amplitudes,
    sweeped_solutions=sweeped_with_total,
    param_text=_format_param_text(
        Q,
        omega_0,
        a,
        b,
        EXCITATION.modal_forces,
        phi_rm=phi_rm,
    ),
    multiples=np.asarray(RESPONSE_MEASURE.multiples),
    mode_labels=mode_labels,
)

plt.show()
