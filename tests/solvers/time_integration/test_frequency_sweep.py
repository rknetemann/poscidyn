import numpy as np
import oscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def F_max (eta, omega_0, Q, gamma):
    return np.sqrt(4 * omega_0**6 / (3 * gamma * Q**2) * (eta + 1 / (2*Q**2)) * (1 + eta + 1 / (4 * Q **2)))

# 1 mode:
Q, omega_0, alpha, gamma = np.array([100.0]), np.array([1.00]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 2.55
modal_forces = np.array([1.0])

# 2 modes:
Q, omega_0, alpha, gamma = np.array([100.0, 100.0]), np.array([1.00, 1.73]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
gamma[0,0,0,0] = 2.55
gamma[0,0,1,1] = 8.61
gamma[1,1,1,1] = 18.7
gamma[1,0,0,1] = 8.57

# 3 modes:
Q, omega_0, alpha, gamma = np.array([100.0, 100.0, 100.0]), np.array([1.00, 1.73, 2.59]), np.zeros((3,3,3)), np.zeros((3,3,3,3))
gamma[0,0,0,0] = 2.55
gamma[0,0,1,1] = 8.61
gamma[0,0,0,2] = 0.942
gamma[0,0,1,2] = -0.143
gamma[0,0,2,2] = 9.88
gamma[1,1,1,1] = 18.7
gamma[1,0,0,1] = 8.57
gamma[1,0,0,2] = -0.000283
gamma[1,0,1,2] = -11.8
gamma[1,0,2,2] = 0.000849
gamma[1,1,1,2] = -0.00272
gamma[1,1,2,2] = 45.2
gamma[2,2,2,2] = -0.00144

F_max_value = F_max(0.20, omega_0[0], Q[0], gamma[0,0,0,0])
print(f"Calculated F_max: {F_max_value:.4f}")

driving_frequency = np.linspace(0.75, 3.0, 601)
driving_amplitude = np.linspace(0.1, 1.0, 5) * F_max_value
modal_forces = np.array([1.0, 1.0, 1.0])

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = oscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(7, 7), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*1, n_time_steps=50, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
SWEEPER = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
PRECISION = oscidyn.Precision.SINGLE

def _extract_gamma_diagonal(gamma: np.ndarray):
    arr = np.asarray(gamma)
    if arr.ndim < 4:
        return arr.ravel().tolist()[: arr.size]

    diag = []
    max_modes = min(2, arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
    for i in range(max_modes):
        diag.append(float(arr[i, i, i, i]))
    return diag


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


def plot_sweep(ax, drive_freqs, drive_amps, sweeped_solutions, param_text: str) -> None:
    forward = sweeped_solutions.get("forward")
    backward = sweeped_solutions.get("backward")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions to plot.")
    
    max_forward = np.max(forward)
    max_backward = np.max(backward)

    max_value = max(
        val for val in [max_forward, max_backward] if val is not None
    )
    print(f"Max value across forward and backward: {max_value}")

    arg_max_forward = np.unravel_index(np.argmax(forward), forward.shape) if forward is not None else None
    arg_max_backward = np.unravel_index(np.argmax(backward), backward.shape) if backward is not None else None

    frequency_at_max_forward = drive_freqs[arg_max_forward[0]] if arg_max_forward is not None else None
    print(f"Max forward at frequency: {frequency_at_max_forward}")

    forward = forward / max_value
    backward = backward / max_value

    gamma_ndim = np.max(max_value**2 * gamma)

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

    ax.set_title(
        f"Frequency sweep\nMax displacement: {max_value:.2f} (omega: {frequency_at_max_forward:.2f})"
    )
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

start_time = time.time()

frequency_sweep = oscidyn.frequency_sweep(
    model = MODEL,
    sweeper=SWEEPER,
    excitor=EXCITOR,
    solver = SOLVER,
    precision = PRECISION,
    multistarter=MULTISTART,
) #n_freq, n_amp, n_init_disp, n_init_vel

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
