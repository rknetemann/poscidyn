import numpy as np
import oscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 1 mode:
Q, omega_0, alpha, gamma = np.array([100.0]), np.array([1.0]), np.zeros((1,1,1)), np.zeros((1,1,1,1))
gamma[0,0,0,0] = 0.0000888

# 2 modes:
# Q, omega_0, alpha, gamma = np.array([30.0, 30.0]), np.array([2.00, 5.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
# gamma[0,0,0,0] = 5.00e-3 * 0.001 * 0
# gamma[1,1,1,1] = 5.00e-3 * 0.001 * 10

REQUIRED_FORCE = 1.0

MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
DRIVING_FREQUENCY = np.linspace(0.1, 2.0, 350)
DRIVING_AMPLITUDE = np.linspace(0.1 * REQUIRED_FORCE, 1.0 * REQUIRED_FORCE, 10)
EXCITOR = oscidyn.OneToneExcitation(drive_frequencies=DRIVING_FREQUENCY, drive_amplitudes=DRIVING_AMPLITUDE, modal_forces=np.array([1.0]))
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(11, 11), linear_response_factor=1.0)
SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*5, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
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
    
    max_forward = np.max(forward)
    max_backward = np.max(backward)

    max_value = max(
        val for val in [max_forward, max_backward] if val is not None
    )
    print(f"Max value across forward and backward: {max_value}")

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
        f"Frequency sweep\nMax displacement: {max_value:.2f}, "
        f"Gamma (non-dimensional): {gamma_ndim:.2e}, "
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
