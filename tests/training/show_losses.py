#!/usr/bin/env python3
"""
Plot train/validation loss from a JSONL log (one JSON object per line).

Usage:
  python plot_losses.py path/to/log.jsonl
  python plot_losses.py path/to/log.jsonl --out losses.png
  python plot_losses.py path/to/log.jsonl --use-global-step

The input format is like:
{"epoch": 1, "global_step": 1250, "train_loss": 0.32, "val_loss": 0.13, ...}
(one JSON object per line)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


def read_jsonl_losses(path: Path) -> Tuple[List[int], List[int], List[float], List[float]]:
    epochs: List[int] = []
    steps: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e

            # Skip lines that don't contain both losses
            if "train_loss" not in obj or "val_loss" not in obj:
                continue

            epoch = obj.get("epoch")
            step = obj.get("global_step")

            if epoch is None and step is None:
                # If neither exists, still allow plotting by index, but keep placeholders.
                epoch = len(epochs) + 1
            if step is None:
                step = len(steps) + 1

            epochs.append(int(epoch))
            steps.append(int(step))
            train_losses.append(float(obj["train_loss"]))
            val_losses.append(float(obj["val_loss"]))

    if not epochs:
        raise ValueError("No valid rows found (expected keys: train_loss and val_loss).")

    return epochs, steps, train_losses, val_losses


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("logfile", type=Path, help="Path to JSONL log file.")
    p.add_argument("--out", type=Path, default=None, help="Optional output image path (e.g., losses.png).")
    p.add_argument("--use-global-step", action="store_true", help="Use global_step for x-axis instead of epoch.")
    p.add_argument("--title", default="Training vs Validation Loss", help="Plot title.")
    args = p.parse_args()

    epochs, steps, train_losses, val_losses = read_jsonl_losses(args.logfile)

    x = steps if args.use_global_step else epochs
    xlabel = "Global step" if args.use_global_step else "Epoch"

    plt.figure()
    plt.plot(x, train_losses, label="Train loss")
    plt.plot(x, val_losses, label="Validation loss")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
