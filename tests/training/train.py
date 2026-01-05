# train.py

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict, field
import json
import csv
import numpy as np
import oscidynn
import jax
from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
import equinox as eqx
import optax

from oscidynn.data.data_normalizer import (
    DatasetNormalizer,
    build_normalizer_from_dataloader,
    save_normalizer,
    try_load_normalizer,
)

DATAFILES = Path(
    "/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/converted"
)

BATCH_SIZE = 64
INIT_LEARNING_RATE = 3e-4
EPOCHS = 50
PRINT_EVERY_EPOCHS = 1
SEED = 5678

# Early stopping
EARLY_STOPPING = True
PATIENCE = 8
MIN_DELTA = 1e-6

# Optional: cap the number of batches per epoch (None = full epoch)
MAX_STEPS_PER_EPOCH: int | None = None

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.eqx"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.eqx"
STATE_PATH = CHECKPOINT_DIR / "train_state.json"
NORMALIZER_PATH = CHECKPOINT_DIR / "normalizer.eqx"

# History
HISTORY_JSONL_PATH = CHECKPOINT_DIR / "history.jsonl"
HISTORY_CSV_PATH = CHECKPOINT_DIR / "history.csv"

# Normalizer build batch size (streaming)
NORMALIZER_BATCH_SIZE = 512

key = jax.random.PRNGKey(SEED)

params = {
    "alpha": [],
    "gamma": [(0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 1, 1), (1, 0, 0, 1)],
}

dataloader = oscidynn.DataLoader(DATAFILES, params=params)


def mean_squared_error(
    y: Float[Array, "batch output_dim"], pred_y: Float[Array, "batch output_dim"]
) -> Float[Array, ""]:
    return jnp.mean((pred_y - y) ** 2)


def loss(
    model: oscidynn.MultiLayerPerceptron,
    normalizer: DatasetNormalizer,
    x: Float[Array, "batch input_dim"],
    y: Float[Array, "batch output_dim"],
) -> Float[Array, ""]:
    x = normalizer.norm_X(x)
    y = normalizer.norm_Y(y)
    pred_y = jax.vmap(model)(x)
    return mean_squared_error(y, pred_y)


loss = eqx.filter_jit(loss)


def r2_score(
    y: Float[Array, "batch output_dim"],
    pred_y: Float[Array, "batch output_dim"],
) -> Float[Array, ""]:
    ss_res = jnp.sum((y - pred_y) ** 2)
    ss_tot = jnp.sum((y - jnp.mean(y)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


@eqx.filter_jit
def evaluate_full_batch(
    model: oscidynn.MultiLayerPerceptron,
    normalizer: DatasetNormalizer,
    x: jnp.ndarray,
    y: jnp.ndarray,
):
    x_n = normalizer.norm_X(x)
    y_n = normalizer.norm_Y(y)
    pred_y_n = jax.vmap(model)(x_n)
    return mean_squared_error(y_n, pred_y_n), r2_score(y_n, pred_y_n)


def batch_index_iterator(rng: np.random.Generator, n: int, batch_size: int):
    indices = np.arange(n)
    while True:
        rng.shuffle(indices)
        for start in range(0, n, batch_size):
            yield indices[start : start + batch_size]


@dataclass
class EarlyStoppingState:
    best_val_loss: float = float("inf")
    best_epoch: int = -1
    epochs_since_improvement: int = 0


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    early_stopping: EarlyStoppingState = field(default_factory=EarlyStoppingState)


def _ensure_checkpoint_dir():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_train_state(state: TrainState) -> None:
    _ensure_checkpoint_dir()
    payload = {
        "epoch": state.epoch,
        "global_step": state.global_step,
        "early_stopping": asdict(state.early_stopping),
    }
    with STATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_train_state() -> TrainState:
    if not STATE_PATH.exists():
        return TrainState()
    with STATE_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    es = payload.get("early_stopping", {})
    return TrainState(
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        early_stopping=EarlyStoppingState(
            best_val_loss=float(es.get("best_val_loss", float("inf"))),
            best_epoch=int(es.get("best_epoch", -1)),
            epochs_since_improvement=int(es.get("epochs_since_improvement", 0)),
        ),
    )


def save_model(path: Path, model: oscidynn.MultiLayerPerceptron) -> None:
    _ensure_checkpoint_dir()
    eqx.tree_serialise_leaves(path, model)


def try_load_model(path: Path, model_template: oscidynn.MultiLayerPerceptron) -> oscidynn.MultiLayerPerceptron | None:
    if not path.exists():
        return None
    return eqx.tree_deserialise_leaves(path, model_template)


# --------------------
# History persistence
# --------------------

@dataclass
class HistoryRow:
    epoch: int
    global_step: int
    train_loss: float
    val_loss: float
    val_r2: float
    best_val_loss: float
    best_epoch: int
    patience_used: int  # epochs_since_improvement
    patience_limit: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_history_jsonl() -> list[HistoryRow]:
    if not HISTORY_JSONL_PATH.exists():
        return []
    rows: list[HistoryRow] = []
    with HISTORY_JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                HistoryRow(
                    epoch=int(obj["epoch"]),
                    global_step=int(obj["global_step"]),
                    train_loss=float(obj["train_loss"]),
                    val_loss=float(obj["val_loss"]),
                    val_r2=float(obj["val_r2"]),
                    best_val_loss=float(obj["best_val_loss"]),
                    best_epoch=int(obj["best_epoch"]),
                    patience_used=int(obj["patience_used"]),
                    patience_limit=int(obj["patience_limit"]),
                )
            )
    return rows


def append_history_jsonl(row: HistoryRow) -> None:
    _ensure_checkpoint_dir()
    with HISTORY_JSONL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row.to_dict()) + "\n")


def write_history_csv(rows: list[HistoryRow]) -> None:
    _ensure_checkpoint_dir()
    fieldnames = list(HistoryRow.__annotations__.keys())
    with HISTORY_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())


def get_or_build_normalizer(*, resume: bool) -> DatasetNormalizer:
    x_dim = int(dataloader.x_dim)
    y_dim = int(dataloader.y_dim)
    template = DatasetNormalizer(
        x_mean=jnp.zeros((1, x_dim), dtype=jnp.float32),
        x_std=jnp.ones((1, x_dim), dtype=jnp.float32),
        y_mean=jnp.zeros((1, y_dim), dtype=jnp.float32),
        y_std=jnp.ones((1, y_dim), dtype=jnp.float32),
    )

    if resume:
        loaded = try_load_normalizer(NORMALIZER_PATH, template)
        if loaded is not None:
            return loaded

    print("Building DatasetNormalizer from training split (streaming)...")
    normalizer = build_normalizer_from_dataloader(
        dataloader,
        split_idx=0,
        batch_size=NORMALIZER_BATCH_SIZE,
        dtype=jnp.float32,
        eps=1e-8,
    )
    _ensure_checkpoint_dir()
    save_normalizer(NORMALIZER_PATH, normalizer)
    print(f"Saved normalizer to '{NORMALIZER_PATH}'.")
    return normalizer


def train(
    model: oscidynn.MultiLayerPerceptron,
    normalizer: DatasetNormalizer,
    optim: optax.GradientTransformation,
    epochs: int,
    batch_size: int,
    print_every_epochs: int,
    *,
    resume: bool = True,
) -> oscidynn.MultiLayerPerceptron:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(SEED)

    n_train = dataloader.n_split_sims[0]
    if n_train is None or n_train <= 0:
        raise RuntimeError("Training split has no simulations.")

    # Validation split as full batch (NumPy -> JAX)
    x_val_np, y_val_np = dataloader.load_data(1, "all")
    x_val = jnp.asarray(x_val_np)
    y_val = jnp.asarray(y_val_np)

    steps_per_epoch = int(np.ceil(n_train / batch_size))
    if MAX_STEPS_PER_EPOCH is not None:
        steps_per_epoch = min(steps_per_epoch, MAX_STEPS_PER_EPOCH)

    batch_iter = batch_index_iterator(rng, n_train, batch_size)

    @eqx.filter_jit
    def make_step(
        model: oscidynn.MultiLayerPerceptron,
        opt_state: PyTree,
        x: Float[Array, "batch input_dim"],
        y: Float[Array, "batch output_dim"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, normalizer, x, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    state = TrainState()
    best_model = model

    history: list[HistoryRow] = []
    if resume:
        history = load_history_jsonl()

        loaded_state = load_train_state()
        maybe_last = try_load_model(LAST_MODEL_PATH, model)
        if maybe_last is not None:
            model = maybe_last
            state = loaded_state
            print(f"Resumed model from '{LAST_MODEL_PATH}'. Starting at epoch {state.epoch + 1}.")
        else:
            state = TrainState()

        maybe_best = try_load_model(BEST_MODEL_PATH, model)
        best_model = maybe_best if maybe_best is not None else model

    es = state.early_stopping

    print("Starting training...")
    for epoch in range(state.epoch + 1, epochs + 1):
        running_loss = 0.0
        n_batches = 0

        for _ in range(steps_per_epoch):
            batch_indices = next(batch_iter)
            x_batch_np, y_batch_np = dataloader.load_data(0, batch_indices.tolist())
            x_batch = jnp.asarray(x_batch_np)
            y_batch = jnp.asarray(y_batch_np)

            model, opt_state, train_loss = make_step(model, opt_state, x_batch, y_batch)

            running_loss += float(train_loss)
            n_batches += 1
            state.global_step += 1

        train_loss_epoch = running_loss / max(1, n_batches)

        val_loss, val_r2 = evaluate_full_batch(model, normalizer, x_val, y_val)
        val_loss_f = float(val_loss)
        val_r2_f = float(val_r2)

        improved = (es.best_val_loss - val_loss_f) > MIN_DELTA
        if improved:
            es.best_val_loss = val_loss_f
            es.best_epoch = epoch
            es.epochs_since_improvement = 0
            best_model = model
            save_model(BEST_MODEL_PATH, best_model)
        else:
            es.epochs_since_improvement += 1

        # Save state + last model
        state.epoch = epoch
        state.early_stopping = es
        save_model(LAST_MODEL_PATH, model)
        save_train_state(state)

        # Record history
        row = HistoryRow(
            epoch=epoch,
            global_step=state.global_step,
            train_loss=float(train_loss_epoch),
            val_loss=val_loss_f,
            val_r2=val_r2_f,
            best_val_loss=float(es.best_val_loss),
            best_epoch=int(es.best_epoch),
            patience_used=int(es.epochs_since_improvement),
            patience_limit=int(PATIENCE),
        )
        history.append(row)
        append_history_jsonl(row)
        write_history_csv(history)

        # Logging
        if (epoch % print_every_epochs) == 0 or (epoch == 1) or (epoch == epochs):
            msg = (
                f"epoch={epoch}/{epochs}, "
                f"train_loss={train_loss_epoch:.6g}, "
                f"val_loss={val_loss_f:.6g}, "
                f"val_r2={val_r2_f:.6g}"
            )
            if EARLY_STOPPING:
                msg += (
                    f", best_val_loss={es.best_val_loss:.6g} (epoch {es.best_epoch}), "
                    f"patience={es.epochs_since_improvement}/{PATIENCE}"
                )
            print(msg)

        if EARLY_STOPPING and es.epochs_since_improvement >= PATIENCE:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val_loss={es.best_val_loss:.6g} at epoch {es.best_epoch}."
            )
            break

    return best_model if EARLY_STOPPING else model


def print_example_prediction(
    model: oscidynn.MultiLayerPerceptron,
    normalizer: DatasetNormalizer,
    *,
    split_idx: int = 1,
    example_idx: int = 0,
    max_items: int = 30,
) -> None:
    x_np, y_np = dataloader.load_data(split_idx, [example_idx])
    x = jnp.asarray(x_np)
    y_true = jnp.asarray(y_np)

    x_n = normalizer.norm_X(x)
    y_pred_n = jax.vmap(model)(x_n)
    y_pred = normalizer.denorm_Y(y_pred_n)

    y_true_1d = np.asarray(y_true[0])
    y_pred_1d = np.asarray(y_pred[0])

    # If gamma was log-transformed for training, exponentiate for human-readable output
    gamma_slice = getattr(dataloader, "y_slices", {}).get("gamma") if hasattr(dataloader, "y_slices") else None
    if getattr(dataloader, "log_gamma", False) and gamma_slice is not None:
        eps = getattr(dataloader, "gamma_log_eps", 0.0)
        y_true_1d = y_true_1d.copy()
        y_pred_1d = y_pred_1d.copy()
        y_true_1d[gamma_slice] = np.exp(y_true_1d[gamma_slice]) - eps
        y_pred_1d[gamma_slice] = np.exp(y_pred_1d[gamma_slice]) - eps

    n = y_true_1d.size
    show = min(max_items, n)

    print("\n--- Example prediction (parameters, denormalized) ---")
    print(f"split_idx={split_idx}, example_idx={example_idx}, n_params={n}")
    for i in range(show):
        print(f"  [{i:03d}] true={y_true_1d[i]: .6e}   pred={y_pred_1d[i]: .6e}")
    if show < n:
        print(f"... ({n - show} more not shown)")
    print("-----------------------------------------------------\n")


# ---- Initialize model from a small batch (NumPy -> JAX) ----
x_init_np, y_init_np = dataloader.load_data(0, [0, 1])
x_init = jnp.asarray(x_init_np)
y_init = jnp.asarray(y_init_np)

key, subkey = jax.random.split(key, 2)
model = oscidynn.MultiLayerPerceptron(
    x_shape=x_init.shape[1],
    y_shape=y_init.shape[1],
    key=subkey,
)

learning_rate = optax.exponential_decay(
    init_value=INIT_LEARNING_RATE,
    decay_rate=0.9,
    transition_steps=100,)

learning_rate_schedule = optax.warmup_cosine_decay_schedule(init_value=INIT_LEARNING_RATE, peak_value=INIT_LEARNING_RATE * 10,
                                                            warmup_steps=500, decay_steps=10000)
optim = optax.adamw(learning_rate_schedule)

try:
    normalizer = get_or_build_normalizer(resume=True)

    model = train(
        model=model,
        normalizer=normalizer,
        optim=optim,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every_epochs=PRINT_EVERY_EPOCHS,
        resume=True,
    )

    print_example_prediction(model, normalizer, split_idx=1, example_idx=0, max_items=30)

finally:
    dataloader.close()
