"""
Utility to inspect simulations stored in a batch_*.hdf5 file.

Use the left/right arrow keys to move through chunks of simulations (default 10 per page).

Example:
    python plot_results.py batch_2025-11-13_10:29:53_0.hdf5 --n-samples 12 --sample-mode random
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a subset of simulations from a batch HDF5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="Path to the batch_*.hdf5 file produced by batch_two_modes.py")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of simulations to plot when --indices is not provided.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("linspace", "random"),
        default="linspace",
        help="Sampling strategy used to pick simulations across the dataset.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Explicit 0-based simulation indices to plot. Overrides --n-samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed used when --sample-mode=random.",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=5,
        help="Maximum number of subplot columns before starting a new row.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save the resulting figure (e.g. out.png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI when saving to --output.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip plt.show(). Useful when running headless and saving to --output.",
    )
    return parser.parse_args()


class SimulationPager:
    """Keeps track of which simulations are shown on each page and enables paging."""

    def __init__(
        self,
        n_available: int,
        page_size: int,
        mode: str,
        explicit: Sequence[int] | None,
        seed: int | None,
    ):
        if n_available == 0:
            raise ValueError("No simulations found in file.")

        self.page_size = max(1, page_size)
        self.random = np.random.default_rng(seed)
        self.mode = mode if explicit is None else "explicit"

        if explicit:
            indices = sorted(set(explicit))
            for idx in indices:
                if idx < 0 or idx >= n_available:
                    raise ValueError(f"Requested simulation index {idx} is outside [0, {n_available - 1}]")
            self.remaining = np.array(indices, dtype=int)
        else:
            self.remaining = np.arange(n_available, dtype=int)

        self.history: list[np.ndarray] = []
        self.current_page = -1
        total_len = len(self.remaining)
        self.total_expected_pages = math.ceil(total_len / self.page_size) if total_len else 0
        self._ensure_page(record=True)

    def _sample_from_remaining(self) -> np.ndarray | None:
        if self.remaining.size == 0:
            return None

        n_pick = min(self.page_size, self.remaining.size)

        if self.mode == "random":
            chosen_idx = self.random.choice(self.remaining.size, size=n_pick, replace=False)
        elif self.mode == "linspace":
            chosen_idx = np.round(
                np.linspace(0, self.remaining.size - 1, n_pick)
            ).astype(int)
        else:  # explicit or sequential fallback
            chosen_idx = np.arange(n_pick)

        chosen_idx = np.sort(chosen_idx)
        selected = self.remaining[chosen_idx]
        self.remaining = np.delete(self.remaining, chosen_idx)
        return selected

    def _ensure_page(self, record: bool) -> bool:
        if self.current_page + 1 < len(self.history):
            if record:
                self.current_page += 1
            return True

        next_page = self._sample_from_remaining()
        if next_page is None:
            return False

        self.history.append(next_page)
        if record:
            self.current_page += 1
        return True

    def advance(self) -> None:
        if self.current_page + 1 < len(self.history):
            self.current_page += 1
            return
        if self._ensure_page(record=True):
            return
        if self.history:
            self.current_page = 0  # wrap to the beginning

    def rewind(self) -> None:
        if self.current_page > 0:
            self.current_page -= 1
            return
        if self.history:
            self.current_page = len(self.history) - 1

    def current_indices(self) -> np.ndarray:
        if not self.history:
            raise RuntimeError("No pages available to display.")
        return self.history[self.current_page]

    def remaining_pages(self) -> int:
        if self.page_size == 0:
            return 0
        total = len(self.history)
        if self.remaining.size:
            total += math.ceil(self.remaining.size / self.page_size)
        return total


def extract_gamma_diagonal(gamma: np.ndarray | None) -> Sequence[float]:
    if gamma is None:
        return []
    arr = np.asarray(gamma)
    if arr.ndim < 4:
        return arr.ravel().tolist()[: arr.size]

    diag = []
    max_modes = min(2, arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
    for i in range(max_modes):
        diag.append(float(arr[i, i, i, i]))
    return diag


def format_param_text(attrs: h5py.AttributeManager) -> str:
    q = np.asarray(attrs.get("Q"))
    omega0 = np.asarray(attrs.get("omega_0"))
    gamma_vals = extract_gamma_diagonal(attrs.get("gamma"))

    parts = []
    if q.size:
        parts.append(f"Q=[{', '.join(f'{val:.2f}' for val in q[:2])}]")
    if omega0.size:
        parts.append(f"omega0=[{', '.join(f'{val:.2f}' for val in omega0[:2])}]")
    if gamma_vals:
        parts.append(f"gamma=[{', '.join(f'{val:.2e}' for val in gamma_vals)}]")
    return "\n".join(parts)


def plot_simulation(ax, sim_group: h5py.Group, sim_label: str) -> None:
    sweeped = np.asarray(sim_group["forward_sweep"])
    f_omegas = np.asarray(sim_group.attrs["f_omegas"])
    f_amps = np.asarray(sim_group.attrs["f_amps"])

    if sweeped.shape[0] != f_omegas.shape[0]:
        raise ValueError(
            f"Frequency axis mismatch for {sim_label}: sweeped shape {sweeped.shape}, "
            f"f_omegas shape {f_omegas.shape}"
        )

    if sweeped.ndim != 2:
        sweeped = sweeped.reshape(sweeped.shape[0], -1)

    if f_amps.ndim == 1:
        amp_labels = f_amps
    else:
        amp_labels = np.linalg.norm(f_amps, axis=1)

    colors = plt.cm.viridis(np.linspace(0, 1, sweeped.shape[1]))
    for amp_idx, (color, response) in enumerate(zip(colors, sweeped.T)):
        ax.plot(
            f_omegas,
            response,
            color=color,
            linewidth=1.1,
            label=f"|F|={amp_labels[amp_idx]:.2f}",
        )

    ax.set_title(sim_label, fontsize=10)
    ax.set_xlabel("Drive frequency")
    ax.set_ylabel("Max displacement")
    ax.grid(alpha=0.2)

    param_text = format_param_text(sim_group.attrs)
    if param_text:
        ax.text(
            0.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6),
        )

    ax.legend(fontsize=6, loc="lower right", frameon=False, ncol=2)


class SimulationBrowser:
    def __init__(
        self,
        args: argparse.Namespace,
        sim_group: h5py.Group,
        sim_keys: list[str],
        pager: SimulationPager,
    ):
        self.args = args
        self.sim_group = sim_group
        self.sim_keys = sim_keys
        self.n_sim = len(sim_keys)
        self.pager = pager
        self.connection_id: int | None = None

        page_slots = max(1, pager.page_size)
        n_cols = min(max(1, args.max_cols), page_slots)
        n_rows = math.ceil(page_slots / n_cols)
        self.fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 4.5, n_rows * 3.2),
            squeeze=False,
            sharey=False,
        )
        self.axes = axes.flatten()
        self._clear_axes()
        self._render_current_page()

    def _clear_axes(self) -> None:
        for ax in self.axes:
            ax.cla()
            ax.axis("off")

    def _render_current_page(self) -> None:
        indices = self.pager.current_indices()
        self._clear_axes()

        for ax, idx in zip(self.axes, indices):
            key = self.sim_keys[idx]
            ax.axis("on")
            label = f"{key} (#{idx})"
            plot_simulation(ax, self.sim_group[key], label)

        page_num = self.pager.current_page + 1
        total_pages = max(page_num, self.pager.remaining_pages())
        self.fig.suptitle(
            f"{self.args.file} - page {page_num}/{total_pages} "
            f"(showing {len(indices)} of {self.n_sim})",
            fontsize=12,
        )
        self.fig.tight_layout(rect=(0, 0, 1, 0.97))
        self.fig.canvas.draw_idle()

    def enable_navigation(self) -> None:
        if self.connection_id is None:
            self.connection_id = self.fig.canvas.mpl_connect("key_press_event", self._on_key)
            print("Keyboard shortcuts: ← previous page, → next page")

    def _on_key(self, event) -> None:
        if event.key == "right":
            self.pager.advance()
            self._render_current_page()
        elif event.key == "left":
            self.pager.rewind()
            self._render_current_page()

    def save(self, path: str, dpi: int) -> None:
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {path}")

    def close(self) -> None:
        plt.close(self.fig)


def main() -> None:
    args = parse_args()

    with h5py.File(args.file, "r") as h5_file:
        if "simulations" not in h5_file:
            raise KeyError("Input file is missing the 'simulations' group.")

        sim_group = h5_file["simulations"]
        sim_keys = sorted(sim_group.keys())
        n_sim = len(sim_keys)
        if n_sim == 0:
            raise ValueError("No simulations stored in file.")

        pager = SimulationPager(
            n_available=n_sim,
            page_size=args.n_samples,
            mode=args.sample_mode,
            explicit=args.indices,
            seed=args.seed,
        )

        browser = SimulationBrowser(args, sim_group, sim_keys, pager)

        if args.output:
            browser.save(args.output, args.dpi)

        if not args.no_show:
            browser.enable_navigation()
            plt.show()
        else:
            browser.close()


if __name__ == "__main__":
    main()
