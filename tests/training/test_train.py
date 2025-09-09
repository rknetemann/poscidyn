import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from pathlib import Path
from typing import Tuple, Iterable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree 
import h5py
import optax
import numpy as np

FILENAME = Path("/home/raymo/Downloads/batch_1.hdf5")
INPUT_SHAPE = (200,)
OUTPUT_DIM = 2
SEED = 42
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
EPOCHS = 200
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

    def __call__(self, x: Float[Array, "200"]) -> Float[Array, "2"]:
        for layer in self.layers:
            x = layer(x)
        return x

def mse_loss(model: MLP, x: Float[Array, "batch 200"], y: Float[Array, "batch 2"]) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)  # (batch, 2)
    return jnp.mean(jnp.sum((preds - y) ** 2, axis=-1))  # MSE over outputs

mse_loss = eqx.filter_jit(mse_loss)

@eqx.filter_jit
def mae_metric(model: MLP, x: Float[Array, "batch 200"], y: Float[Array, "batch 2"]) -> Float[Array, ""]:
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

def _stack_group_datasets(grp: h5py.Group) -> np.ndarray:
    # Stack all dataset children along axis 0
    arrs = []
    for name, obj in grp.items():
        if isinstance(obj, h5py.Dataset):
            arrs.append(obj[...])
    if not arrs:
        raise ValueError("Group contains no datasets to stack.")
    return np.concatenate(arrs, axis=0)

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

def load_hdf5_xy(filename: Path) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    HDF5 layout:
      - group 'simulations' with datasets for each simulation.
      - each dataset is one simulation consisting of multiple sweeps:
          shape (200, 10)   OR (10, 200)   OR flat (2000,) == 10*200
      - each dataset has attrs: 'Q' (float) and 'gamma' (float).
      - optional:
          'driving_frequencies': (n_freqs,)  -> used to infer n_freqs (default 200)
          'driving_amplitudes':  (n_sweeps,) -> used to sanity-check sweep count
    Returns:
      X: (N, 200) float32
      Y: (N, 2)   float32, columns [Q, gamma]
    """
    with h5py.File(filename, "r") as f:
        if "simulations" not in f or not isinstance(f["simulations"], h5py.Group):
            raise ValueError("Expected a 'simulations' group in the HDF5 file.")
        sims_grp: h5py.Group = f["simulations"]

        # infer n_freqs from metadata, else default to 200
        if "driving_frequencies" in f:
            freqs = np.asarray(f["driving_frequencies"][...])
            n_freqs = int(freqs.shape[0])
        else:
            n_freqs = INPUT_SHAPE[0]  # 200

        # optional amplitudes metadata (only for sanity checks)
        amps = None
        if "driving_amplitudes" in f:
            amps = np.asarray(f["driving_amplitudes"][...])
            n_sweeps_meta = int(amps.shape[0])
        else:
            n_sweeps_meta = None

        sim_names = [nm for nm, obj in sims_grp.items() if isinstance(obj, h5py.Dataset)]
        sim_names.sort()

        X_list, Y_list = [], []

        for nm in sim_names:
            ds = sims_grp[nm]
            raw = ds[...]
            # Parse data into (n_sweeps, n_freqs)
            if raw.ndim == 2 and raw.shape[0] == n_freqs:      # (200, S)
                sweeps = raw.T
            elif raw.ndim == 2 and raw.shape[1] == n_freqs:    # (S, 200)
                sweeps = raw
            elif raw.ndim == 1 and (raw.size % n_freqs == 0):  # (S*200,)
                sweeps = raw.reshape(-1, n_freqs)
            else:
                raise ValueError(
                    f"Simulation '{nm}' has shape {raw.shape}; "
                    f"expected (n_freqs, S), (S, n_freqs), or flat (S*n_freqs,)."
                )
            
            sweeps = sweeps.astype(np.float32)
            normalize_rows_inplace(sweeps, use_abs=True, eps=1e-8)   # <<< normalize each sweep

            n_sweeps = sweeps.shape[0]
            if n_sweeps_meta is not None and n_sweeps != n_sweeps_meta:
                print(f"Warning: '{nm}' has {n_sweeps} sweeps; metadata amplitudes imply {n_sweeps_meta}.")

            # ---- targets from attributes ----
            if "Q" not in ds.attrs or "gamma" not in ds.attrs:
                raise ValueError(f"Dataset '{nm}' missing 'Q' and/or 'gamma' attrs.")
            Q = float(ds.attrs["Q"])
            gamma = float(ds.attrs["gamma"])

            # Repeat [Q, gamma] for each sweep in this dataset
            Y_sim = np.tile(np.array([Q, gamma], dtype=np.float32), (n_sweeps, 1))

            X_list.append(sweeps.astype(np.float32))
            Y_list.append(Y_sim)

        X = np.concatenate(X_list, axis=0)  # (total_samples, n_freqs)
        Y = np.concatenate(Y_list, axis=0)  # (total_samples, 2)

        # final checks
        if X.ndim != 2 or X.shape[1] != INPUT_SHAPE[0]:
            raise ValueError(f"X must be (N, {INPUT_SHAPE[0]}). Got {X.shape}")
        if Y.ndim != 2 or Y.shape[1] != 2 or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be (N, 2) with same N as X. Got {Y.shape}, X={X.shape}")

        return jnp.asarray(X, dtype=jnp.float32), jnp.asarray(Y, dtype=jnp.float32)

def train_test_split(X: jnp.ndarray, Y: jnp.ndarray, test_frac=0.2, seed=SEED):
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_test = int(N * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]

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
def evaluate(model: MLP, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int = 1024) -> Tuple[float, float]:
    total_mse, total_mae, total_n = 0.0, 0.0, 0
    for xb, yb in batch_iter(X, Y, batch_size, shuffle=False, seed=0):
        bsz = xb.shape[0]
        total_mse += float(mse_loss(model, xb, yb)) * bsz
        total_mae += float(mae_metric(model, xb, yb)) * bsz
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
          epochs: int, batch_size: int, print_every: int) -> MLP:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    step = 0
    for epoch in range(1, epochs+1):
        for xb, yb in batch_iter(Xtr, Ytr, batch_size, shuffle=True, seed=SEED+epoch):
            model, opt_state, train_loss = train_step(model, opt_state, xb, yb)
            if (step % print_every) == 0:
                te_mse, te_mae = evaluate(model, Xte, Yte)
                print(f"epoch={epoch:02d} step={step:05d}  train_mse={float(train_loss):.6f}  test_mse={te_mse:.6f}  test_mae={te_mae:.6f}")
            step += 1
    # final metrics
    te_mse, te_mae = evaluate(model, Xte, Yte)
    print(f"[final] test_mse={te_mse:.6f}  test_mae={te_mae:.6f}")
    return model

# ----------------------
# Demo / run
# ----------------------
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
model = MLP(subkey)

# Quick sanity check on shapes with random data
test_x = jax.random.normal(key, (28, 200))
test_y = jax.random.normal(key, (28, 2))
pred_y = jax.vmap(model)(test_x)
print("Predicted y (first sample):", pred_y[0])
value, grads = eqx.filter_value_and_grad(mse_loss)(model, test_x, test_y)
print("Initial MSE (random):", float(value))

# Load your actual data
try:
    X, Y = load_hdf5_xy(FILENAME)
except Exception as e:
    # Helpful fallback so the script still runs; remove once your file layout is recognized
    print(f"Warning: {e}\nUsing synthetic data as a fallback.")
    N = 5000
    X = jax.random.normal(key, (N, 200))
    # Some synthetic regression target
    W = jax.random.normal(key, (200, 2))
    Y = jnp.tanh(X @ W) + 0.05 * jax.random.normal(key, (N, 2))

Xtr, Ytr, Xte, Yte = train_test_split(X, Y, test_frac=0.2, seed=SEED)

# Train
model = train(model, Xtr, Ytr, Xte, Yte, epochs=EPOCHS, batch_size=BATCH_SIZE, print_every=PRINT_EVERY)
