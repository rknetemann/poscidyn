"""Load a saved test_train model and reproduce evaluation plots without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np

from test_train import (
    BATCH_SIZE,
    FILENAME,
    MODEL_STATE_PATH,
    SEED,
    TrainingArtifacts,
    evaluate,
    load_hdf5_xy,
    load_training_state,
    plot_prediction_stats,
    print_sample_predictions,
    simulation_train_test_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot predictions from a saved test_train model without retraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=MODEL_STATE_PATH,
        help="Path to the serialized TrainingArtifacts file produced by test_train.py",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=FILENAME,
        help="Dataset used for evaluation (same format as training).",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of simulations reserved for evaluation when rebuilding the split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed used both for the template deserialization and the train/test split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size used when computing evaluation metrics.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="How many random samples to print as predicted vs actual pairs.",
    )
    parser.add_argument(
        "--q-filter",
        type=float,
        nargs=2,
        metavar=("Q1", "Q2"),
        help="Restrict plots/metrics to simulations whose (Q1, Q2) match these values.",
    )
    parser.add_argument(
        "--q-filter-tol",
        type=float,
        default=1e-6,
        help="Tolerance used when matching Q1/Q2 for --q-filter.",
    )
    parser.add_argument(
        "--omega02-max",
        type=float,
        help="Keep only samples whose second natural frequency satisfies omega_0_2 < value.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib plots (metrics and sample prints still run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.state_path.expanduser()
    if not state_path.exists():
        raise FileNotFoundError(
            f"State file '{state_path}' does not exist. Run test_train.py to generate it first."
        )

    artifacts: TrainingArtifacts = load_training_state(state_path, jax.random.PRNGKey(args.seed))

    X_np, Y_np, sim_ids = load_hdf5_xy(args.hdf5)
    Xtr_np, Ytr_np, Xte_np, Yte_np = simulation_train_test_split(
        X_np, Y_np, sim_ids, test_frac=args.test_frac, seed=args.seed
    )

    if args.q_filter is not None:
        q1_req, q2_req = args.q_filter
        tol = args.q_filter_tol
        mask = np.isclose(Yte_np[:, 0], q1_req, atol=tol, rtol=0.0) & np.isclose(
            Yte_np[:, 1], q2_req, atol=tol, rtol=0.0
        )
        if not np.any(mask):
            raise ValueError(
                f"No evaluation samples matched Q1={q1_req} and Q2={q2_req} within tol={tol}."
            )
        Xte_np = Xte_np[mask]
        Yte_np = Yte_np[mask]
        print(
            f"Filtered evaluation set to {Xte_np.shape[0]} samples with Q1≈{q1_req}, Q2≈{q2_req}."
        )

    if args.omega02_max is not None:
        freq_mask = Yte_np[:, 3] < args.omega02_max
        if not np.any(freq_mask):
            raise ValueError(
                f"No evaluation samples had omega_0_2 < {args.omega02_max}."
            )
        Xte_np = Xte_np[freq_mask]
        Yte_np = Yte_np[freq_mask]
        print(
            f"Filtered evaluation set to {Xte_np.shape[0]} samples with omega_0_2 < {args.omega02_max}."
        )

    Xte = artifacts.normalizer.norm_X(Xte_np)
    Yte = artifacts.normalizer.norm_Y(Yte_np)

    mse, mae = evaluate(
        artifacts.model,
        Xte,
        Yte,
        artifacts.normalizer,
        batch_size=args.batch_size,
    )
    print(f"Evaluation MSE={mse:.6f}  MAE={mae:.6f}")

    print_sample_predictions(
        artifacts.model,
        Xte,
        Yte,
        artifacts.normalizer,
        n_samples=args.n_samples,
        seed=args.seed,
    )

    if args.no_plots:
        return

    preds_all = np.array(artifacts.normalizer.denorm_Y(jax.vmap(artifacts.model)(Xte)))
    truths_all = np.array(artifacts.normalizer.denorm_Y(Yte))
    plot_prediction_stats(preds_all, truths_all)


if __name__ == "__main__":
    main()
