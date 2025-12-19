#!/usr/bin/env python3
# evaluate_model.py

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

import oscidynn  # your package
from oscidynn.data.data_normalizer import DatasetNormalizer  # your module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved Equinox model on a split + plots.")
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to folder containing HDF5 simulation files (same as training DATAFILES).",
    )
    p.add_argument(
        "--checkpoints",
        type=Path,
        default=Path("checkpoints"),
        help="Checkpoint directory containing best_model.eqx and normalizer.eqx",
    )
    p.add_argument(
        "--which",
        choices=["best", "last"],
        default="best",
        help="Which model checkpoint to load.",
    )
    p.add_argument(
        "--split",
        type=int,
        default=2,
        help="Which split to evaluate on: 0=train, 1=val, 2=test (default=2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Inference batch size (controls memory use).",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="How many example parameter-vectors to plot/print.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("eval_outputs"),
        help="Output directory for plots.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=5678,
        help="Seed used only for selecting random examples to plot.",
    )

    # New: per-parameter predicted vs true plots
    p.add_argument(
        "--per-param-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate predicted-vs-true plots for every parameter dimension (default: True).",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Max points per per-parameter plot (random downsample if more).",
    )
    return p.parse_args()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_model_and_normalizer(
    dataloader: oscidynn.DataLoader,
    checkpoints_dir: Path,
    which: str,
    seed: int,
):
    model_path = checkpoints_dir / ("best_model.eqx" if which == "best" else "last_model.eqx")
    norm_path = checkpoints_dir / "normalizer.eqx"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalizer checkpoint not found: {norm_path}")

    # Build template model using dims from one small batch
    x_init_np, y_init_np = dataloader.load_data(0, [0, 1])
    x_init = jnp.asarray(x_init_np)
    y_init = jnp.asarray(y_init_np)

    key = jax.random.PRNGKey(seed)
    _, subkey = jax.random.split(key, 2)

    model_template = oscidynn.MultiLayerPerceptron(
        x_shape=x_init.shape[1],
        y_shape=y_init.shape[1],
        key=subkey,
    )
    model = eqx.tree_deserialise_leaves(model_path, model_template)

    # Template normalizer with correct shapes
    x_dim = int(dataloader.x_dim)
    y_dim = int(dataloader.y_dim)
    norm_template = DatasetNormalizer(
        x_mean=jnp.zeros((1, x_dim), dtype=jnp.float32),
        x_std=jnp.ones((1, x_dim), dtype=jnp.float32),
        y_mean=jnp.zeros((1, y_dim), dtype=jnp.float32),
        y_std=jnp.ones((1, y_dim), dtype=jnp.float32),
    )
    normalizer = eqx.tree_deserialise_leaves(norm_path, norm_template)
    return model, normalizer


@eqx.filter_jit
def predict_batch(model, normalizer, x: jnp.ndarray) -> jnp.ndarray:
    x_n = normalizer.norm_X(x)
    y_pred_n = jax.vmap(model)(x_n)
    return y_pred_n


def streamed_predict(
    dataloader: oscidynn.DataLoader,
    model,
    normalizer: DatasetNormalizer,
    split_idx: int,
    batch_size: int,
):
    n = dataloader.n_split_sims[split_idx]
    if n is None or n <= 0:
        raise RuntimeError(f"Split {split_idx} has no simulations.")

    y_true_all = []
    y_pred_all = []

    for start in range(0, n, batch_size):
        idxs = list(range(start, min(start + batch_size, n)))
        x_np, y_np = dataloader.load_data(split_idx, idxs)
        x = jnp.asarray(x_np)
        y_true = jnp.asarray(y_np)

        y_pred_n = predict_batch(model, normalizer, x)
        y_pred = normalizer.denorm_Y(y_pred_n)

        # Move to CPU NumPy
        y_true_all.append(np.asarray(y_true))
        y_pred_all.append(np.asarray(y_pred))

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    return y_true_all, y_pred_all


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def r2_score(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def per_dim_mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y_pred - y_true), axis=0)


def plot_scatter_flat(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path) -> None:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    n = yt.size
    if n > 200_000:
        idx = np.random.default_rng(0).choice(n, size=200_000, replace=False)
        yt = yt[idx]
        yp = yp[idx]

    plt.figure()
    plt.scatter(yt, yp, s=4, alpha=0.25)
    mn = min(np.min(yt), np.min(yp))
    mx = max(np.max(yt), np.max(yp))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True parameters (all dims, flattened)")
    plt.ylabel("Predicted parameters (all dims, flattened)")
    plt.title("Predicted vs True (flattened)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_residual_hist(y_true: np.ndarray, y_pred: np.ndarray, outpath: Path) -> None:
    resid = (y_pred - y_true).reshape(-1)
    if resid.size > 0:
        lo, hi = np.quantile(resid, [0.005, 0.995])
        resid = resid[(resid >= lo) & (resid <= hi)]

    plt.figure()
    plt.hist(resid, bins=80)
    plt.xlabel("Residual (pred - true), clipped to 0.5%..99.5%")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_topk_mae(mae: np.ndarray, outpath: Path, k: int = 30) -> None:
    k = min(k, mae.size)
    idx = np.argsort(mae)[::-1][:k]
    vals = mae[idx]

    plt.figure()
    plt.bar(np.arange(k), vals)
    plt.xticks(np.arange(k), [str(i) for i in idx], rotation=90)
    plt.xlabel("Parameter dimension index (top-K by MAE)")
    plt.ylabel("MAE (original units)")
    plt.title(f"Top-{k} parameter dimensions by MAE")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_example_param_vectors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    example_indices: list[int],
    outpath: Path,
    max_dims: int = 200,
) -> None:
    y_dim = y_true.shape[1]
    d = min(y_dim, max_dims)

    plt.figure()
    for i in example_indices:
        plt.plot(np.arange(d), y_true[i, :d], linewidth=1.0, label=f"true[{i}]")
        plt.plot(np.arange(d), y_pred[i, :d], linewidth=1.0, linestyle="--", label=f"pred[{i}]")
    plt.xlabel("Parameter dimension index")
    plt.ylabel("Value (original units)")
    title = "Example parameter vectors (true vs pred)"
    if d < y_dim:
        title += f" (first {d}/{y_dim} dims)"
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_example_sweeps(
    dataloader: oscidynn.DataLoader,
    split_idx: int,
    example_indices: list[int],
    outpath: Path,
    max_points: int = 2000,
) -> None:
    plt.figure()
    for i in example_indices:
        x_np, _ = dataloader.load_data(split_idx, [i])
        sweep = x_np[0]
        if sweep.size > max_points:
            idx = np.linspace(0, sweep.size - 1, max_points).astype(int)
            sweep = sweep[idx]
        plt.plot(sweep, linewidth=1.0, label=f"sweep[{i}]")
    plt.xlabel("Sweep index")
    plt.ylabel("Amplitude (as stored)")
    plt.title("Example input sweeps")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pred_vs_true_per_param(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    *,
    max_points: int = 20000,
    seed: int = 0,
) -> None:
    """
    Saves one figure per parameter dimension:
      scatter(true_j, pred_j) with identity line y=x.

    Files:
      out_dir/param_0000.png, param_0001.png, ...

    If there are many samples, randomly downsample to max_points for speed/readability.
    """
    ensure_dir(out_dir)
    rng = np.random.default_rng(seed)

    n, y_dim = y_true.shape
    if n != y_pred.shape[0] or y_dim != y_pred.shape[1]:
        raise ValueError("y_true and y_pred shape mismatch.")

    # Prepare optional downsample indices once (same subset for all params)
    if n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        yt = y_true[idx, :]
        yp = y_pred[idx, :]
        n_plot = max_points
    else:
        yt = y_true
        yp = y_pred
        n_plot = n

    for j in range(y_dim):
        xj = yt[:, j]
        yj = yp[:, j]

        # If this dimension is constant (std=0), scatter is uninformative; still save.
        mn = float(min(np.min(xj), np.min(yj)))
        mx = float(max(np.max(xj), np.max(yj)))
        if mn == mx:
            mn -= 0.5
            mx += 0.5

        plt.figure()
        plt.scatter(xj, yj, s=6, alpha=0.35)
        plt.plot([mn, mx], [mn, mx])
        plt.xlabel(f"True (param {j})")
        plt.ylabel(f"Predicted (param {j})")
        plt.title(f"Predicted vs True for parameter {j} (n={n_plot})")
        plt.tight_layout()
        plt.savefig(out_dir / f"param_{j:04d}.png", dpi=200)
        plt.close()


def main():
    args = parse_args()
    ensure_dir(args.out)

    # Keep params selection consistent with training (adjust if needed)
    params = {
        "alpha": [],
        "gamma": [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)],
    }

    dataloader = oscidynn.DataLoader(args.data, params=params)

    try:
        model, normalizer = load_model_and_normalizer(
            dataloader=dataloader,
            checkpoints_dir=args.checkpoints,
            which=args.which,
            seed=args.seed,
        )

        # Predict on split
        y_true, y_pred = streamed_predict(
            dataloader=dataloader,
            model=model,
            normalizer=normalizer,
            split_idx=args.split,
            batch_size=args.batch_size,
        )

        # Metrics
        overall_mse = mse(y_true, y_pred)
        overall_r2 = r2_score(y_true, y_pred)
        mae = per_dim_mae(y_true, y_pred)

        print("\n=== Evaluation ===")
        print(f"split_idx: {args.split}")
        print(f"N samples: {y_true.shape[0]}")
        print(f"y_dim:     {y_true.shape[1]}")
        print(f"MSE (denorm/original units): {overall_mse:.6e}")
        print(f"R2  (denorm/original units): {overall_r2:.6f}")
        print(f"Mean MAE:   {float(np.mean(mae)):.6e}")
        print(f"Median MAE: {float(np.median(mae)):.6e}")

        # Examples
        rng = np.random.default_rng(args.seed)
        n = y_true.shape[0]
        ex = rng.choice(n, size=min(args.max_examples, n), replace=False).tolist()

        print("\n--- Example predicted vs true parameters (first 20 dims) ---")
        for i in ex:
            t = y_true[i, :20]
            p = y_pred[i, :20]
            print(f"example {i}:")
            for j in range(t.size):
                print(f"  [{j:02d}] true={t[j]: .6e}   pred={p[j]: .6e}")

        # Summary plots
        plot_scatter_flat(y_true, y_pred, args.out / "scatter_pred_vs_true_flat.png")
        plot_residual_hist(y_true, y_pred, args.out / "residual_hist.png")
        plot_topk_mae(mae, args.out / "topk_mae.png", k=30)
        plot_example_param_vectors(y_true, y_pred, ex, args.out / "example_param_vectors.png", max_dims=200)
        plot_example_sweeps(dataloader, args.split, ex, args.out / "example_sweeps.png", max_points=2000)

        # Per-parameter plots (requested)
        if args.per_param_plots:
            per_param_dir = args.out / "per_param"
            print(f"\nGenerating per-parameter predicted-vs-true plots into: {per_param_dir.resolve()}")
            plot_pred_vs_true_per_param(
                y_true,
                y_pred,
                per_param_dir,
                max_points=args.max_points,
                seed=args.seed,
            )
            print(f"Done. Created {y_true.shape[1]} per-parameter plots.")

        print(f"\nSaved outputs to: {args.out.resolve()}\n")

    finally:
        dataloader.close()


if __name__ == "__main__":
    main()
