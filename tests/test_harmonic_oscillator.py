import numpy as np
import poscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

zeta = np.array([0.01, 0.02])
omega_0 = np.array([1.0, 2.0])

driving_frequency = np.linspace(0.5, 2.0, 500)
driving_amplitude = np.linspace(0.1, 1.0, 10)
modal_forces = np.array([1.0, 1.0])

MODEL = poscidyn.HarmonicOscillator(zeta=zeta, omega_0=omega_0)
EXCITATION = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*5, n_time_steps=50, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
SWEEPER = poscidyn.NearestNeighbourSweep(sweep_direction=[poscidyn.Forward(), poscidyn.Backward()])
PRECISION = poscidyn.Precision.SINGLE


def plot_sweep(ax, drive_freqs, drive_amps, sweeped_solutions) -> None:
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

    ax.set_title(
        f"Frequency sweep"
    )
    ax.set_xlabel("Drive frequency")
    ax.set_ylabel("Max displacement")
    ax.grid(alpha=0.25)

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

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL,
    sweeper=SWEEPER,
    excitation=EXCITATION,
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
    drive_freqs=EXCITATION.drive_frequencies,
    drive_amps=EXCITATION.drive_amplitudes,
    sweeped_solutions=frequency_sweep.sweeped_periodic_solutions,
)
plt.tight_layout()
plt.show()
