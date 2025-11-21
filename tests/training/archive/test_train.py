import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from pathlib import Path
from typing import Iterable, Tuple

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HDF5 = REPO_ROOT / "batch_2025-11-13_12:34:28_0.hdf5"

def _resolve_hdf5_files(defaults: Iterable[Path]) -> list[Path]:
    """
    Allows specifying multiple training files via OSC_TRAIN_HDF5 (os.pathsep-separated).
    Falls back to the provided defaults if the env var is not set.
    """
    env_value = os.environ.get("OSC_TRAIN_HDF5")
    if env_value:
        candidates = [Path(part.strip()) for part in env_value.split(os.pathsep) if part.strip()]
        if candidates:
            return candidates
    return [Path(p) for p in defaults]

# Add or remove entries here to train on multiple batches when no env var is set.
DEFAULT_HDF5_FILES = [
    Path("/home/raymo/Downloads/2025-11-20/batch_0_2025-11-20_17-26.hdf5"), 
    Path("/home/raymo/Downloads/2025-11-20/batch_1_2025-11-20_17-27.hdf5"),
    Path("/home/raymo/Downloads/2025-11-20/batch_2_2025-11-20_17-27.hdf5"),
    Path("/home/raymo/Downloads/2025-11-20/batch_3_2025-11-20_17-37.hdf5")
]
HDF5_FILES = _resolve_hdf5_files(DEFAULT_HDF5_FILES or [DEFAULT_HDF5])
DEFAULT_STATE_PATH = REPO_ROOT / "results" / "test_train_state.eqx"
MODEL_STATE_PATH = Path(os.environ.get("OSC_TRAIN_STATE", DEFAULT_STATE_PATH))
N_FREQS = 200  # single-force sweep contains 200 frequency samples
INPUT_DIM = N_FREQS  # each sweep row flattens to 200 scalars
TRAIN_SWEEP_DIRECTIONS = ("forward", "backward")  # choose subset of ("forward", "backward")
OUTPUT_DIM = 6  # [Q1, Q2, omega01, omega02, gamma1, gamma2]
SEED = 42
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
EPOCHS = 150
PRINT_EVERY = 100  # steps

# ----------------------
# Model
# ----------------------
class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(INPUT_DIM, 128, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 128, key=k2),
            jax.nn.relu,
            eqx.nn.Linear(128, 64, key=k3),
            jax.nn.relu,
            eqx.nn.Linear(64, OUTPUT_DIM, key=k4),
        ]

    def __call__(self, x: Float[Array, "input_dim"]) -> Float[Array, "6"]:
        for layer in self.layers:
            x = layer(x)
        return x

def mse_loss(model: MLP, x: Float[Array, "batch input_dim"], y: Float[Array, "batch 6"]) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)  # (batch, OUTPUT_DIM)
    return jnp.mean(jnp.sum((preds - y) ** 2, axis=-1))  # MSE over outputs

mse_loss = eqx.filter_jit(mse_loss)

@eqx.filter_jit
def mae_metric(model: MLP, x: Float[Array, "batch input_dim"], y: Float[Array, "batch 6"]) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)
    return jnp.mean(jnp.abs(preds - y))

# ----------------------
# Data loading helpers
# ----------------------
def _as_jnp(x) -> jnp.ndarray:
    # ensure float32 for JAX speed
    x = np.asarray(x)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32)
    elif x.dtype != np.float32:
        x = x.astype(np.float32)
    return jnp.array(x)

def _pad_or_trim(vec: np.ndarray, length: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((length,), fill, dtype=np.float32)
    if vec.size:
        take = min(length, vec.size)
        out[:take] = vec[:take]
    return out

def extract_gamma_diagonal(gamma_attr) -> np.ndarray:
    if gamma_attr is None:
        return np.asarray([], dtype=np.float32)
    arr = np.asarray(gamma_attr, dtype=np.float32)
    if arr.ndim < 4:
        return arr.reshape(-1)
    n = min(arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
    diag = [arr[i, i, i, i] for i in range(n)]
    return np.asarray(diag, dtype=np.float32)

def _prepare_sweep_matrix(raw: np.ndarray, n_freqs: int) -> np.ndarray:
    """
    Normalizes arbitrary layouts into (n_sweeps, n_freqs).
    Accepts arrays shaped (n_freqs, n_amp), (n_amp, n_freqs), flat, or higher-rank with n_freqs on axis 0/last.
    """
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[0] == n_freqs:
        sweeps = arr.T
    elif arr.ndim == 2 and arr.shape[1] == n_freqs:
        sweeps = arr
    elif arr.ndim >= 3 and arr.shape[0] == n_freqs:
        sweeps = arr.reshape(n_freqs, -1).T
    elif arr.ndim >= 3 and arr.shape[-1] == n_freqs:
        sweeps = arr.reshape(-1, n_freqs)
    elif arr.ndim == 1 and (arr.size % n_freqs == 0):
        sweeps = arr.reshape(-1, n_freqs)
    else:
        raise ValueError(
            f"Unexpected sweep shape {arr.shape}; unable to infer samples with n_freqs={n_freqs}."
        )
    return sweeps.astype(np.float32)

def _build_target_vector(attrs: h5py.AttributeManager, gamma_override=None) -> np.ndarray:
    if "Q" not in attrs or "omega_0" not in attrs:
        raise ValueError("Each simulation must provide 'Q' and 'omega_0' attributes.")
    q_vals = np.asarray(attrs["Q"], dtype=np.float32).reshape(-1)
    omega_vals = np.asarray(attrs["omega_0"], dtype=np.float32).reshape(-1)
    if gamma_override is None:
        if "gamma" not in attrs:
            raise ValueError("Missing 'gamma' attribute for targets.")
        gamma_source = attrs["gamma"]
    else:
        gamma_source = gamma_override
    gamma_diag = extract_gamma_diagonal(gamma_source)
    target = np.concatenate([
        _pad_or_trim(q_vals, 2),
        _pad_or_trim(omega_vals, 2),
        _pad_or_trim(gamma_diag, 2),
    ])
    if target.shape[0] != OUTPUT_DIM:
        raise ValueError(f"Constructed target has shape {target.shape}; expected length {OUTPUT_DIM}.")
    return target

class DatasetNormalizer(eqx.Module):
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    @classmethod
    def from_data(cls, X: np.ndarray, Y: np.ndarray, eps: float = 1e-8) -> "DatasetNormalizer":
        x_mean = jnp.asarray(X.mean(axis=0, keepdims=True))
        x_std = jnp.asarray(X.std(axis=0, keepdims=True) + eps)
        y_mean = jnp.asarray(Y.mean(axis=0, keepdims=True))
        y_std = jnp.asarray(Y.std(axis=0, keepdims=True) + eps)
        return cls(x_mean, x_std, y_mean, y_std)

    def norm_X(self, X: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(X) - self.x_mean) / self.x_std

    def norm_Y(self, Y: np.ndarray | jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(Y) - self.y_mean) / self.y_std

    def denorm_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        return Y * self.y_std + self.y_mean


class TrainingArtifacts(eqx.Module):
    model: MLP
    normalizer: DatasetNormalizer


def _empty_normalizer() -> DatasetNormalizer:
    zeros_x = jnp.zeros((1, INPUT_DIM), dtype=jnp.float32)
    zeros_y = jnp.zeros((1, OUTPUT_DIM), dtype=jnp.float32)
    return DatasetNormalizer(
        x_mean=zeros_x,
        x_std=jnp.ones_like(zeros_x),
        y_mean=zeros_y,
        y_std=jnp.ones_like(zeros_y),
    )


def create_artifact_template(key: jax.random.PRNGKey) -> TrainingArtifacts:
    return TrainingArtifacts(model=MLP(key), normalizer=_empty_normalizer())


def save_training_state(path: Path, artifacts: TrainingArtifacts) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, artifacts)


def load_training_state(path: Path, key: jax.random.PRNGKey) -> TrainingArtifacts:
    template = create_artifact_template(key)
    return eqx.tree_deserialise_leaves(path, template)


def _load_single_hdf5_file(filename: Path, sweep_dataset_names: list[str], n_freqs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HDF5 layout per file:
      - group 'simulations' with directional sweeps ('forward_sweep'/'backward_sweep').
      - each sweep dataset stores normalized amplitudes (n_freqs, n_forces) and has gamma_ndim attrs.
      - simulation attributes include 'Q' and 'omega_0'.
    Returns data for a single file.
    """
    with h5py.File(filename, "r") as f:
        if "simulations" not in f or not isinstance(f["simulations"], h5py.Group):
            raise ValueError(f"Expected a 'simulations' group in the HDF5 file '{filename}'.")
        sims_grp: h5py.Group = f["simulations"]

        sim_names = sorted(sims_grp.keys())
        if not sim_names:
            raise ValueError(f"No simulations found in HDF5 file '{filename}'.")
        X_list, Y_list, sim_id_list = [], [], []

        for sim_idx, nm in enumerate(sim_names):
            sim_grp = sims_grp[nm]
            if not isinstance(sim_grp, h5py.Group):
                raise TypeError(
                    f"Simulation '{nm}' must be stored as a group with directional sweeps. "
                    f"Found {type(sim_grp)}."
                )

            samples_added = 0
            for sweep_name in sweep_dataset_names:
                if sweep_name not in sim_grp:
                    continue
                dataset = sim_grp[sweep_name]
                sweeps = _prepare_sweep_matrix(dataset[...], n_freqs)
                if sweeps.shape[1] != n_freqs:
                    raise ValueError(
                        f"Sweep '{sweep_name}' in simulation '{nm}' has width {sweeps.shape[1]}; expected {n_freqs}."
                    )
                sweeps = sweeps.astype(np.float32)
                target_vec = _build_target_vector(sim_grp.attrs, gamma_override=dataset.attrs.get("gamma_ndim"))
                Y_sim = np.repeat(target_vec[None, :], sweeps.shape[0], axis=0)

                X_list.append(sweeps)
                Y_list.append(Y_sim)
                sim_id_list.append(np.full((sweeps.shape[0],), sim_idx, dtype=np.int32))
                samples_added += sweeps.shape[0]

            if samples_added == 0:
                raise ValueError(
                    f"Simulation '{nm}' in file '{filename}' did not contain any of {sweep_dataset_names}."
                )

        X = np.concatenate(X_list, axis=0)  # (total_samples, input_dim)
        Y = np.concatenate(Y_list, axis=0)  # (total_samples, OUTPUT_DIM)
        sim_ids = np.concatenate(sim_id_list, axis=0)

        if X.ndim != 2 or X.shape[1] != INPUT_DIM:
            raise ValueError(f"X must be (N, {INPUT_DIM}). Got {X.shape} from '{filename}'")
        if Y.ndim != 2 or Y.shape[1] != OUTPUT_DIM or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be (N, {OUTPUT_DIM}) matching X. Got {Y.shape} from '{filename}'")

        return X.astype(np.float32), Y.astype(np.float32), sim_ids


def load_hdf5_xy(filenames: Iterable[Path] | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Provides concatenated training data from one or more HDF5 files.
    """
    if isinstance(filenames, (str, os.PathLike, Path)):
        file_list = [Path(filenames)]
    else:
        file_list = [Path(f) for f in filenames]
    if not file_list:
        raise ValueError("No HDF5 files specified.")
    if not TRAIN_SWEEP_DIRECTIONS:
        raise ValueError("TRAIN_SWEEP_DIRECTIONS must contain at least one sweep direction.")
    valid_dirs = {"forward", "backward"}
    invalid = [d for d in TRAIN_SWEEP_DIRECTIONS if d not in valid_dirs]
    if invalid:
        raise ValueError(f"Invalid sweep direction(s): {invalid}. Allowed: {sorted(valid_dirs)}")
    sweep_dataset_names = [f"{direction}_sweep" for direction in TRAIN_SWEEP_DIRECTIONS]

    X_parts, Y_parts, sim_id_parts = [], [], []
    sim_offset = 0
    for path in file_list:
        Xf, Yf, sim_ids_f = _load_single_hdf5_file(path, sweep_dataset_names, N_FREQS)
        X_parts.append(Xf)
        Y_parts.append(Yf)
        sim_id_parts.append(sim_ids_f + sim_offset)
        sim_offset += int(sim_ids_f.max()) + 1

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    sim_ids = np.concatenate(sim_id_parts, axis=0)
    return X, Y, sim_ids

def simulation_train_test_split(
    X: np.ndarray,
    Y: np.ndarray,
    sim_ids: np.ndarray,
    test_frac: float = 0.2,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_sims = np.unique(sim_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_sims)
    n_test = max(1, int(np.round(len(unique_sims) * test_frac)))
    test_sims = set(unique_sims[:n_test])
    test_mask = np.array([sid in test_sims for sid in sim_ids])
    train_mask = ~test_mask
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("Simulation split failed; adjust test_frac or dataset size.")
    return X[train_mask], Y[train_mask], X[test_mask], Y[test_mask]

def batch_iter(X: jnp.ndarray, Y: jnp.ndarray, batch_size: int, shuffle: bool, seed: int) -> Iterable[Tuple[jnp.ndarray, jnp.ndarray]]:
    N = X.shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, N, batch_size):
        sl = idx[start:start+batch_size]
        yield X[sl], Y[sl]

# ----------------------
# Evaluation utilities
# ----------------------
@eqx.filter_jit
def predict_denorm(model: MLP, X: jnp.ndarray, normalizer: DatasetNormalizer) -> jnp.ndarray:
    preds = jax.vmap(model)(X)
    return normalizer.denorm_Y(preds)


@eqx.filter_jit
def evaluate(model: MLP, X: jnp.ndarray, Y: jnp.ndarray, normalizer: DatasetNormalizer) -> Tuple[jnp.ndarray, jnp.ndarray]:
    preds_raw = predict_denorm(model, X, normalizer)
    y_raw = normalizer.denorm_Y(Y)
    diff = preds_raw - y_raw
    mse = jnp.mean(jnp.sum(diff ** 2, axis=-1))
    mae = jnp.mean(jnp.abs(diff))
    return mse, mae

# ----------------------
# Train step
# ----------------------
optim = optax.adamw(LEARNING_RATE)

@eqx.filter_jit
def train_step(model: MLP, opt_state: PyTree, x: jnp.ndarray, y: jnp.ndarray):
    loss_value, grads = eqx.filter_value_and_grad(mse_loss)(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value

def train(model: MLP, Xtr: jnp.ndarray, Ytr: jnp.ndarray, Xte: jnp.ndarray, Yte: jnp.ndarray,
          normalizer: DatasetNormalizer, epochs: int, batch_size: int, print_every: int) -> MLP:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    step = 0
    for epoch in range(1, epochs+1):
        for xb, yb in batch_iter(Xtr, Ytr, batch_size, shuffle=True, seed=SEED+epoch):
            model, opt_state, train_loss = train_step(model, opt_state, xb, yb)
            if (step % print_every) == 0:
                tr_mse, tr_mae = evaluate(model, Xtr, Ytr, normalizer)
                te_mse, te_mae = evaluate(model, Xte, Yte, normalizer)
                print(
                    f"epoch={epoch:02d} step={step:05d}  "
                    f"train_mse={float(tr_mse):.6f}  train_mae={float(tr_mae):.6f}  "
                    f"test_mse={float(te_mse):.6f}  test_mae={float(te_mae):.6f}"
                )
            step += 1
    # final metrics
    te_mse, te_mae = evaluate(model, Xte, Yte, normalizer)
    print(f"[final] test_mse={float(te_mse):.6f}  test_mae={float(te_mae):.6f}")
    return model

def print_sample_predictions(
    model: MLP,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    normalizer: DatasetNormalizer,
    n_samples: int = 5,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = min(n_samples, X.shape[0])
    indices = rng.choice(X.shape[0], size=n_samples, replace=False)
    xb = X[indices]
    yb = Y[indices]
    preds_raw = np.array(predict_denorm(model, xb, normalizer))
    yb_raw = np.array(normalizer.denorm_Y(yb))
    for i, (pred_vec, true_vec) in enumerate(zip(preds_raw, yb_raw), start=1):
        print(f"\n[Example {i}]")
        print(f"  predicted = {pred_vec}")
        print(f"  actual    = {true_vec}")
        diff = pred_vec - true_vec
        print(f"  abs diff  = {np.abs(diff)}")

    return preds_raw, yb_raw


def plot_prediction_stats(preds: np.ndarray, truths: np.ndarray) -> None:
    if plt is None:
        print("matplotlib not available; skipping plots.")
        return
    target_labels = ["Q1", "Q2", "omega01", "omega02", "gamma1", "gamma2"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= preds.shape[1]:
            ax.axis("off")
            continue
        ax.scatter(truths[:, i], preds[:, i], s=10, alpha=0.5)
        lim_min = float(min(truths[:, i].min(), preds[:, i].min()))
        lim_max = float(max(truths[:, i].max(), preds[:, i].max()))
        lim_range = lim_max - lim_min
        if lim_range < 1e-6:
            center = 0.5 * (lim_max + lim_min)
            span = max(np.max(np.abs(preds[:, i] - truths[:, i])) * 3.0, 1e-6)
            lower = center - span
            upper = center + span
        else:
            margin = max(lim_range * 0.1, 1e-6)
            lower = lim_min - margin
            upper = lim_max + margin
        ax.plot([lower, upper], [lower, upper], "r--", linewidth=1)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(target_labels[i])
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        mae = np.mean(np.abs(preds[:, i] - truths[:, i]))
        rmse = np.sqrt(np.mean((preds[:, i] - truths[:, i]) ** 2))
        ax.text(
            0.05,
            0.95,
            f"MAE={mae:.2e}\nRMSE={rmse:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        ax.grid(alpha=0.2)
    fig.tight_layout()

    # Residual histograms for another view
    fig_err, axes_err = plt.subplots(2, 3, figsize=(12, 6), sharex=False)
    axes_err = axes_err.flatten()
    for i, ax in enumerate(axes_err):
        if i >= preds.shape[1]:
            ax.axis("off")
            continue
        residuals = preds[:, i] - truths[:, i]
        residuals = residuals[np.isfinite(residuals)]
        ax.set_title(f"{target_labels[i]} residuals")
        ax.set_xlabel("Pred - Actual")
        ax.set_ylabel("Count")
        if residuals.size == 0:
            ax.text(0.5, 0.5, "no finite data", ha="center", va="center", transform=ax.transAxes)
        elif np.ptp(residuals) < 1e-12:
            ax.axvline(float(np.mean(residuals)), color="tab:blue", linewidth=2)
            ax.text(0.5, 0.8, "all residuals equal", ha="center", transform=ax.transAxes, fontsize=8)
            ax.set_xlim(float(residuals[0]) - 1e-6, float(residuals[0]) + 1e-6)
        else:
            n_bins = min(40, max(5, int(np.ceil(np.sqrt(residuals.size)))))
            ax.hist(residuals, bins=n_bins, color="tab:blue", alpha=0.8)
        ax.axvline(0.0, color="r", linestyle="--", linewidth=1)
        ax.grid(alpha=0.2)
    fig_err.tight_layout()
    plt.show()
    
def main() -> None:
    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)
    model = MLP(subkey)

    # Quick sanity check on shapes with random data
    test_x = jax.random.normal(key, (28, INPUT_DIM))
    test_y = jax.random.normal(key, (28, OUTPUT_DIM))
    pred_y = jax.vmap(model)(test_x)
    print("Predicted y (first sample):", pred_y[0])
    value, _ = eqx.filter_value_and_grad(mse_loss)(model, test_x, test_y)
    print("Initial MSE (random):", float(value))

    # Load your actual data
    try:
        X_np, Y_np, sim_ids = load_hdf5_xy(HDF5_FILES)
    except Exception as e:
        # Helpful fallback so the script still runs; remove once your file layout is recognized
        print(f"Warning: {e}\nUsing synthetic data as a fallback.")
        N = 5000
        X = jax.random.normal(key, (N, INPUT_DIM))
        # Some synthetic regression target
        W = jax.random.normal(key, (INPUT_DIM, OUTPUT_DIM))
        Y = jnp.tanh(X @ W) + 0.05 * jax.random.normal(key, (N, OUTPUT_DIM))
        sim_ids = np.arange(N) // 50
        X_np = np.array(X, dtype=np.float32)
        Y_np = np.array(Y, dtype=np.float32)

    Xtr_np, Ytr_np, Xte_np, Yte_np = simulation_train_test_split(
        X_np, Y_np, sim_ids, test_frac=0.2, seed=SEED
    )
    normalizer = DatasetNormalizer.from_data(Xtr_np, Ytr_np)
    Xtr = normalizer.norm_X(Xtr_np)
    Ytr = normalizer.norm_Y(Ytr_np)
    Xte = normalizer.norm_X(Xte_np)
    Yte = normalizer.norm_Y(Yte_np)

    # Train
    model = train(
        model,
        Xtr,
        Ytr,
        Xte,
        Yte,
        normalizer,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        print_every=PRINT_EVERY,
    )
    print_sample_predictions(model, Xte, Yte, normalizer, n_samples=5, seed=SEED)
    if plt is not None:
        preds_all = np.array(predict_denorm(model, Xte, normalizer))
        truths_all = np.array(normalizer.denorm_Y(Yte))
        plot_prediction_stats(preds_all, truths_all)

    artifacts = TrainingArtifacts(model=model, normalizer=normalizer)
    save_training_state(MODEL_STATE_PATH, artifacts)
    print(f"Saved trained model state to {MODEL_STATE_PATH}")


if __name__ == "__main__":
    main()
