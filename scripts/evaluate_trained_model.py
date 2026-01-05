"""
Evaluate a trained model checkpoint:
  - loads model + normalizer
  - computes MSE/R2 on a chosen split
  - prints a few example predictions
  - plots/saves loss curves from history.jsonl

Defaults mirror tests/training/train.py (dataset path and params).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

import oscidynn
from oscidynn.data.data_loader import DataLoader
from oscidynn.data.data_normalizer import DatasetNormalizer
from typing import Sequence


DEFAULT_DATA = Path("/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/converted")
DEFAULT_PARAMS = {
    "alpha": [],
    "gamma": [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)],
}


def r2_score(y_true: jnp.ndarray, y_pred: jnp.ndarray) -> float:
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[warn] history file not found: {path}")
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def plot_history(rows: list[dict], out_path: Path, show: bool = False) -> None:
    if not rows:
        print("[warn] no history to plot")
        return
    epochs = [r["epoch"] for r in rows]
    train_losses = [r["train_loss"] for r in rows]
    val_losses = [r["val_loss"] for r in rows]

    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[info] saved loss plot to {out_path}")
    if show:
        plt.show()
    plt.close()


def build_normalizer_template(dataloader: DataLoader) -> DatasetNormalizer:
    x_dim = int(dataloader.x_dim)
    y_dim = int(dataloader.y_dim)
    return DatasetNormalizer(
        x_mean=jnp.zeros((1, x_dim), dtype=jnp.float32),
        x_std=jnp.ones((1, x_dim), dtype=jnp.float32),
        y_mean=jnp.zeros((1, y_dim), dtype=jnp.float32),
        y_std=jnp.ones((1, y_dim), dtype=jnp.float32),
    )


def load_normalizer(path: Path, dataloader: DataLoader) -> DatasetNormalizer:
    template = build_normalizer_template(dataloader)
    if not path.exists():
        raise FileNotFoundError(f"Normalizer not found at {path}")
    return eqx.tree_deserialise_leaves(path, template)


def load_model(path: Path, dataloader: DataLoader, key: jax.Array) -> oscidynn.MultiLayerPerceptron:
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    template = oscidynn.MultiLayerPerceptron(
        x_shape=int(dataloader.x_dim),
        y_shape=int(dataloader.y_dim),
        key=key,
    )
    return eqx.tree_deserialise_leaves(path, template)


def maybe_unlog_gamma(arr: np.ndarray, dataloader: DataLoader) -> np.ndarray:
    gamma_slice = getattr(dataloader, "y_slices", {}).get("gamma") if hasattr(dataloader, "y_slices") else None
    if getattr(dataloader, "log_gamma", False) and gamma_slice is not None:
        eps = getattr(dataloader, "gamma_log_eps", 0.0)
        arr = arr.copy()
        arr[..., gamma_slice] = np.exp(arr[..., gamma_slice]) - eps
    return arr


def evaluate_split(
    dataloader: DataLoader,
    normalizer: DatasetNormalizer,
    model: oscidynn.MultiLayerPerceptron,
    split_idx: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    x_np, y_np = dataloader.load_data(split_idx, "all")
    x = jnp.asarray(x_np)
    y = jnp.asarray(y_np)

    x_n = normalizer.norm_X(x)
    y_n = normalizer.norm_Y(y)
    pred_n = jax.vmap(model)(x_n)

    # Metrics in normalized space (matches training loss)
    mse_n = float(jnp.mean((pred_n - y_n) ** 2))
    r2_n = r2_score(y_n, pred_n)

    # Metrics in data space
    pred = normalizer.denorm_Y(pred_n)
    mse_raw = float(jnp.mean((pred - y) ** 2))
    r2_raw = r2_score(y, pred)

    metrics = {
        "mse_normalized": mse_n,
        "r2_normalized": r2_n,
        "mse_raw": mse_raw,
        "r2_raw": r2_raw,
        "n_samples": x.shape[0],
    }

    # For display, optionally undo log transform on gamma
    y_disp = maybe_unlog_gamma(np.asarray(y), dataloader)
    pred_disp = maybe_unlog_gamma(np.asarray(pred), dataloader)

    return metrics, y_disp, pred_disp


def print_examples(y_true: np.ndarray, y_pred: np.ndarray, max_examples: int, max_dims: int, split_idx: int) -> None:
    n_examples = min(max_examples, y_true.shape[0])
    n_dims = min(max_dims, y_true.shape[1])
    print(f"\n--- Example predictions (split {split_idx}) ---")
    for i in range(n_examples):
        print(f"[example {i}]")
        for j in range(n_dims):
            print(f"  [{j:03d}] true={y_true[i, j]: .6e}   pred={y_pred[i, j]: .6e}")
        remaining_dims = y_true.shape[1] - n_dims
        if remaining_dims > 0:
            print(f"  ... ({remaining_dims} more dims not shown)")
    print("-------------------------------------------------\n")


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
        if include_backward:
            bwd_ds = file["backward_sweeps"][sim_name]
            bwd = np.asarray(bwd_ds[...]).reshape(-1)

        sweeps.append(
            {
                "sim_name": sim_name,
                "freqs": freqs,
                "forward": fwd,
                "backward": bwd,
            }
        )
    return sweeps


def plot_sweeps(sweeps, out_path: Path, show: bool = False) -> None:
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
        ax.set_title(item["sim_name"])
        ax.set_xlabel("scaled_f_omega")
        ax.set_ylabel("response")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[info] saved sweep plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained model checkpoint and visualize loss.")
    p.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"), help="Directory with model/normalizer/history.")
    p.add_argument("--history", type=Path, default=None, help="Path to history.jsonl (defaults to checkpoint_dir/history.jsonl).")
    p.add_argument("--model", type=Path, default=None, help="Path to model file (defaults to checkpoint_dir/best_model.eqx).")
    p.add_argument("--normalizer", type=Path, default=None, help="Path to normalizer file (defaults to checkpoint_dir/normalizer.eqx).")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to converted dataset (file or directory).")
    p.add_argument("--split", type=int, default=1, help="Split index to evaluate (0=train,1=val,2=test...).")
    p.add_argument("--examples", type=int, default=3, help="How many example items to print.")
    p.add_argument("--example-dims", type=int, default=12, help="How many parameters to show per example.")
    p.add_argument("--show-plot", action="store_true", help="Show the loss plot in addition to saving it.")
    args = p.parse_args()

    ckpt_dir = args.checkpoint_dir
    model_path = args.model or (ckpt_dir / "best_model.eqx")
    norm_path = args.normalizer or (ckpt_dir / "normalizer.eqx")
    history_path = args.history or (ckpt_dir / "history.jsonl")

    dataloader = DataLoader(
        args.data,
        params=DEFAULT_PARAMS,
        log_gamma=True,
        gamma_log_eps=1e-12,
    )
    try:
        normalizer = load_normalizer(norm_path, dataloader)
        model = load_model(model_path, dataloader, key=jax.random.PRNGKey(0))

        metrics, y_true, y_pred = evaluate_split(dataloader, normalizer, model, args.split)
        print(f"[info] evaluated split {args.split} on {metrics['n_samples']} samples")
        for k, v in metrics.items():
            if k == "n_samples":
                continue
            print(f"  {k}: {v:.6g}")

        print_examples(y_true, y_pred, args.examples, args.example_dims, args.split)

        history = load_history(history_path)
        if history:
            plot_history(history, ckpt_dir / "loss_curve.png", show=args.show_plot)

        # Sweep visualization for the same examples we printed
        example_indices = list(range(min(args.examples, y_true.shape[0])))
        sweep_examples = collect_sweeps(dataloader, args.split, example_indices, include_backward=True)
        plot_sweeps(sweep_examples, ckpt_dir / f"sweeps_split{args.split}.png", show=args.show_plot)
    finally:
        dataloader.close()


if __name__ == "__main__":
    main()
