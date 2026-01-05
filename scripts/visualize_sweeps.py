"""
Plot forward/backward frequency sweeps for selected simulations, showing the sweep
difference (L1 distance between sweeps) in the title.

Defaults mirror tests/training/train.py for data location and params.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from oscidynn.data.data_loader import DataLoader

DEFAULT_DATA = Path("/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/converted")
DEFAULT_PARAMS = {
    "alpha": [],
    "gamma": [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)],
}


def collect_sweeps(
    dataloader: DataLoader,
    split_idx: int,
    example_indices: Sequence[int],
    include_backward: bool = True,
):
    """Fetch forward/backward sweeps and frequency axis for given split indices."""
    sweeps = []
    opened = dataloader._get_or_open_files()  # uses cached handles
    for idx in example_indices:
        file_idx, file_sim_idx = dataloader.splits_sims_idxs[split_idx][idx]
        file, sim_names = opened[file_idx]
        sim_name = sim_names[file_sim_idx]

        fwd_ds = file["forward_sweeps"][sim_name]
        freqs = np.asarray(fwd_ds.attrs["scaled_f_omegas"]).reshape(-1)
        fwd = np.asarray(fwd_ds[...]).reshape(-1)

        bwd = None
        sweep_diff = None
        if include_backward:
            bwd_ds = file["backward_sweeps"][sim_name]
            bwd = np.asarray(bwd_ds[...]).reshape(-1)
            sweep_diff = float(np.sum(np.abs(fwd - bwd)))

        sweeps.append(
            {
                "sim_name": sim_name,
                "freqs": freqs,
                "forward": fwd,
                "backward": bwd,
                "sweep_difference": sweep_diff,
            }
        )
    return sweeps


def plot_sweeps(sweeps, out_path: Path | None, show: bool = False) -> None:
    if not sweeps:
        print("[warn] no sweeps to plot")
        return
    n = len(sweeps)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, item in zip(axes, sweeps):
        ax.plot(item["freqs"], item["forward"], label="forward", lw=1.5)
        if item["backward"] is not None:
            ax.plot(item["freqs"], item["backward"], label="backward", lw=1.0, alpha=0.8)
        title = item["sim_name"]
        if item["sweep_difference"] is not None:
            title += f" | sweep_diff={item['sweep_difference']:.3e}"
        ax.set_title(title)
        ax.set_xlabel("scaled_f_omega")
        ax.set_ylabel("response")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
        print(f"[info] saved sweep plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualize forward/backward sweeps with sweep difference in title.")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to converted dataset (file or directory).")
    p.add_argument("--split", type=int, default=1, help="Split index to draw from (0=train,1=val,2=test...).")
    p.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Split-relative simulation indices to plot.",
    )
    p.add_argument("--output", type=Path, default=Path("results/sweeps.png"), help="Where to save the plot.")
    p.add_argument("--show", action="store_true", help="Show plot interactively.")
    args = p.parse_args()

    dataloader = DataLoader(
        args.data,
        params=DEFAULT_PARAMS,
        log_gamma=True,
        gamma_log_eps=1e-12,
    )
    try:
        # Clamp requested indices to available sims for convenience
        max_idx = dataloader.n_split_sims[args.split] - 1
        filtered_indices = [i for i in args.indices if 0 <= i <= max_idx]
        if not filtered_indices:
            raise ValueError(f"No valid indices to load (requested {args.indices}, max available {max_idx}).")

        sweeps = collect_sweeps(dataloader, args.split, filtered_indices, include_backward=True)
        plot_sweeps(sweeps, args.output, show=args.show)
    finally:
        dataloader.close()


if __name__ == "__main__":
    main()
