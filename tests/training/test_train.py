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
# FILENAME = Path(os.environ.get("OSC_TRAIN_HDF5", DEFAULT_HDF5))
FILENAME = Path("/home/raymo/Downloads/batch_0_2025-11-13_15-54-52.hdf5") #Newer
#FILENAME = Path("/home/raymo/Downloads/batch_0_2025-11-13_11-38-54.hdf5") #Older
FILENAME = Path("/home/raymo/Downloads/batch_0_2025-11-17_17-03.hdf5")
DEFAULT_STATE_PATH = REPO_ROOT / "results" / "test_train_state.eqx"
MODEL_STATE_PATH = Path(os.environ.get("OSC_TRAIN_STATE", DEFAULT_STATE_PATH))
N_FREQS = 300
EXTRA_FEATURES = 3  # max amplitude per sweep + min/max driving frequency
INPUT_SHAPE = (N_FREQS + EXTRA_FEATURES,)
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
            eqx.nn.Linear(INPUT_SHAPE[0], 128, key=k1),
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

    def __call__(self, x: Float[Array, "303"]) -> Float[Array, "6"]:
        for layer in self.layers:
            x = layer(x)
        return x

def mse_loss(model: MLP, x: Float[Array, "batch 303"], y: Float[Array, "batch 6"]) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)  # (batch, OUTPUT_DIM)
    return jnp.mean(jnp.sum((preds - y) ** 2, axis=-1))  # MSE over outputs

mse_loss = eqx.filter_jit(mse_loss)

@eqx.filter_jit
def mae_metric(model: MLP, x: Float[Array, "batch 303"], y: Float[Array, "batch 6"]) -> Float[Array, ""]:
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

def normalize_rows_inplace(a: np.ndarray, use_abs: bool = True, eps: float = 1e-8) -> None:
    """
    Normalizes each row of 'a' by its max (optionally abs max) value.
    Operates in-place. Rows whose max <= eps are left unchanged.
    """
    if use_abs:
        row_max = np.max(np.abs(a), axis=1, keepdims=True)
    else:
        row_max = np.max(a, axis=1, keepdims=True)
    scale = np.maximum(row_max, eps)
    a /= scale

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

def _build_target_vector(attrs: h5py.AttributeManager) -> np.ndarray:
    if "Q" not in attrs or "gamma" not in attrs or "omega_0" not in attrs:
        raise ValueError("Each simulation must provide 'Q', 'omega_0', and 'gamma' attributes.")
    q_vals = np.asarray(attrs["Q"], dtype=np.float32).reshape(-1)
    omega_vals = np.asarray(attrs["omega_0"], dtype=np.float32).reshape(-1)
    gamma_diag = extract_gamma_diagonal(attrs["gamma"])
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
    zeros_x = jnp.zeros((1, INPUT_SHAPE[0]), dtype=jnp.float32)
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


def load_hdf5_xy(filename: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HDF5 layout:
      - group 'simulations' with either datasets (legacy layout) or nested groups per simulation.
      - new layout stores datasets such as 'sweeped_periodic_solutions' and 'x_max_total' inside each simulation group.
      - each simulation provides attrs: 'Q' (length-2), 'omega_0' (length-2), 'gamma' (>=2 diag entries).
    Returns:
      X: (N, INPUT_SHAPE[0]) float32, i.e. sweep samples + max amplitude channel
      Y: (N, 6)   float32, columns [Q1, Q2, omega01, omega02, gamma1, gamma2]
    """
    with h5py.File(filename, "r") as f:
        if "simulations" not in f or not isinstance(f["simulations"], h5py.Group):
            raise ValueError("Expected a 'simulations' group in the HDF5 file.")
        sims_grp: h5py.Group = f["simulations"]

        n_freqs = N_FREQS
        sim_names = sorted(sims_grp.keys())
        if not sim_names:
            raise ValueError("No simulations found in the provided HDF5 file.")
        X_list, Y_list, sim_id_list = [], [], []

        for sim_idx, nm in enumerate(sim_names):
            obj = sims_grp[nm]

            if isinstance(obj, h5py.Dataset):
                sweeps = _prepare_sweep_matrix(obj[...], n_freqs)
                attrs = obj.attrs
            elif isinstance(obj, h5py.Group):
                if "sweeped_periodic_solutions" in obj:
                    raw = obj["sweeped_periodic_solutions"][...]
                elif "x_max_total" in obj:
                    raw = np.asarray(obj["x_max_total"][...])
                    raw = np.nanmax(raw, axis=(-2, -1))  # collapse seed axes -> (n_freqs, n_amp)
                else:
                    raise ValueError(
                        f"Simulation '{nm}' missing a recognizable dataset "
                        "(expected 'sweeped_periodic_solutions' or 'x_max_total')."
                    )
                sweeps = _prepare_sweep_matrix(raw, n_freqs)
                attrs = obj.attrs
            else:
                raise TypeError(f"Unsupported object type {type(obj)} for '{nm}'.")

            row_scale = np.max(np.abs(sweeps), axis=1, keepdims=True)
            row_scale = np.maximum(row_scale, 1e-8)
            scale_feature = np.log1p(row_scale)
            if "f_omegas" not in attrs:
                raise ValueError(f"Simulation '{nm}' missing 'f_omegas' attribute for feature construction.")
            freqs = np.asarray(attrs["f_omegas"], dtype=np.float32).reshape(-1)
            freq_pair = np.array([float(np.min(freqs)), float(np.max(freqs))], dtype=np.float32)
            freq_features = np.repeat(freq_pair[None, :], sweeps.shape[0], axis=0)

            normalize_rows_inplace(sweeps, use_abs=True, eps=1e-8)
            features = np.concatenate(
                [sweeps.astype(np.float32), scale_feature.astype(np.float32), freq_features.astype(np.float32)],
                axis=1,
            )
            if features.shape[1] != INPUT_SHAPE[0]:
                raise ValueError(f"Expected input width {INPUT_SHAPE[0]}, got {features.shape[1]} for '{nm}'.")

            target_vec = _build_target_vector(attrs)
            Y_sim = np.repeat(target_vec[None, :], features.shape[0], axis=0)

            X_list.append(features)
            Y_list.append(Y_sim)
            sim_id_list.append(np.full((features.shape[0],), sim_idx, dtype=np.int32))

        X = np.concatenate(X_list, axis=0)  # (total_samples, input_dim)
        Y = np.concatenate(Y_list, axis=0)  # (total_samples, OUTPUT_DIM)
        sim_ids = np.concatenate(sim_id_list, axis=0)

        # final checks
        if X.ndim != 2 or X.shape[1] != INPUT_SHAPE[0]:
            raise ValueError(f"X must be (N, {INPUT_SHAPE[0]}). Got {X.shape}")
        if Y.ndim != 2 or Y.shape[1] != OUTPUT_DIM or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be (N, {OUTPUT_DIM}) with same N as X. Got {Y.shape}, X={X.shape}")

        return X.astype(np.float32), Y.astype(np.float32), sim_ids

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
# Evaluation
# ----------------------
def evaluate(model: MLP, X: jnp.ndarray, Y: jnp.ndarray, normalizer: DatasetNormalizer, batch_size: int = 1024) -> Tuple[float, float]:
    total_mse, total_mae, total_n = 0.0, 0.0, 0
    for xb, yb in batch_iter(X, Y, batch_size, shuffle=False, seed=0):
        bsz = xb.shape[0]
        preds = jax.vmap(model)(xb)
        preds_raw = normalizer.denorm_Y(preds)
        yb_raw = normalizer.denorm_Y(yb)
        diff = preds_raw - yb_raw
        batch_mse = jnp.mean(jnp.sum(diff ** 2, axis=-1))
        batch_mae = jnp.mean(jnp.abs(diff))
        total_mse += float(batch_mse) * bsz
        total_mae += float(batch_mae) * bsz
        total_n += bsz
    return total_mse / total_n, total_mae / total_n

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
                tr_mse, tr_mae = evaluate(model, Xtr, Ytr, normalizer, batch_size=batch_size)
                te_mse, te_mae = evaluate(model, Xte, Yte, normalizer, batch_size=batch_size)
                print(
                    f"epoch={epoch:02d} step={step:05d}  "
                    f"train_mse={tr_mse:.6f}  train_mae={tr_mae:.6f}  "
                    f"test_mse={te_mse:.6f}  test_mae={te_mae:.6f}"
                )
            step += 1
    # final metrics
    te_mse, te_mae = evaluate(model, Xte, Yte, normalizer, batch_size=batch_size)
    print(f"[final] test_mse={te_mse:.6f}  test_mae={te_mae:.6f}")
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
    preds = jax.vmap(model)(xb)
    preds_raw = np.array(normalizer.denorm_Y(preds))
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
    test_x = jax.random.normal(key, (28, INPUT_SHAPE[0]))
    test_y = jax.random.normal(key, (28, OUTPUT_DIM))
    pred_y = jax.vmap(model)(test_x)
    print("Predicted y (first sample):", pred_y[0])
    value, _ = eqx.filter_value_and_grad(mse_loss)(model, test_x, test_y)
    print("Initial MSE (random):", float(value))

    # Load your actual data
    try:
        X_np, Y_np, sim_ids = load_hdf5_xy(FILENAME)
    except Exception as e:
        # Helpful fallback so the script still runs; remove once your file layout is recognized
        print(f"Warning: {e}\nUsing synthetic data as a fallback.")
        N = 5000
        X = jax.random.normal(key, (N, INPUT_SHAPE[0]))
        # Some synthetic regression target
        W = jax.random.normal(key, (INPUT_SHAPE[0], OUTPUT_DIM))
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
        preds_all = np.array(normalizer.denorm_Y(jax.vmap(model)(Xte)))
        truths_all = np.array(normalizer.denorm_Y(Yte))
        plot_prediction_stats(preds_all, truths_all)

    artifacts = TrainingArtifacts(model=model, normalizer=normalizer)
    save_training_state(MODEL_STATE_PATH, artifacts)
    print(f"Saved trained model state to {MODEL_STATE_PATH}")


if __name__ == "__main__":
    main()
