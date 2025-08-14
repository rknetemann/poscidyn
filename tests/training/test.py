"""
Train a simple JAX MLP to learn max steady‑state displacement
for Duffing oscillator sweeps.

Features per sample: [Q, gamma, sweep_direction, driving_frequency, driving_amplitude]
Target per sample:   max_steady_state_displacement (scalar)

The script expects an HDF5 file structured like:
- dataset "driving_frequencies": shape (n_f,), 1D
- dataset "driving_amplitudes":  shape (n_a,), 1D
- group   "simulations": contains multiple sims. For each sim:
    - attrs: Q (float), gamma (float), sweep_direction (str|int|float)
    - data:  max_steady_state_displacement with shape (n_f * n_a,) or (n_f, n_a)

Outputs:
- Trains an MLP (Flax + Optax) with MSE loss.
- Prints validation RMSE/MAE.
- Provides helper function `predict_grid(...)` to reconstruct a (n_f, n_a) grid.

Requires: jax, flax, optax, h5py, numpy
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import h5py
import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax

# ----------------------------
# Configuration
# ----------------------------
FILENAME = "/home/raymo/Projects/parameter-identification-nanomechanical-resonators/batch_2025-08-14_15:16:59_0.hdf5"  # <- change if needed
RNG_SEED = 0
BATCH_SIZE = 8192
EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZES = (128, 128, 64)
VALID_FRACTION = 0.2
PRINT_EVERY = 5

# Hold out an entire simulation for parameter estimation/validation
HOLDOUT_ONE_SIM = True
HOLDOUT_SIM_INDEX = 'middle'  # 'middle' or an integer index

# ----------------------------
# Utilities
# ----------------------------

def encode_sweep_direction(sd) -> float:
    """Map sweep_direction attribute to a numeric value.
    Accepts strings like 'up'/'down' or bytes, or numeric types.
    """
    if isinstance(sd, (bytes, bytearray)):
        try:
            sd = sd.decode("utf-8").strip().lower()
        except Exception:
            sd = str(sd)
    if isinstance(sd, str):
        s = sd.strip().lower()
        if s in ("up", "forward", "+", "increasing"):
            return 1.0
        if s in ("down", "backward", "-", "decreasing"):
            return -1.0
        # fallback: try to parse float
        try:
            return float(s)
        except Exception:
            return 0.0
    try:
        return float(sd)
    except Exception:
        return 0.0

@dataclass
class Normalizer:
    X_mean: np.ndarray
    X_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def norm_X(self, X: np.ndarray) -> np.ndarray:
        return (X - self.X_mean) / self.X_std

    def denorm_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean

# ----------------------------
# Data loading
# ----------------------------

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Normalizer, Dict]:
    """Load HDF5 file and produce (X_train, y_train, X_val, y_val, normalizer, aux).

    X columns: [Q, gamma, sweep_dir, freq, amp]
    y: scalar max steady-state displacement

    aux contains: {
        'freqs': 1D np.ndarray,
        'amps':  1D np.ndarray,
        'per_sim_meta': list of dicts for each simulation with Q/gamma/sweep_dir
    }
    """
    with h5py.File(filename, "r") as hdf5:
        freqs = np.array(hdf5["driving_frequencies"][:], dtype=np.float32)
        amps  = np.array(hdf5["driving_amplitudes"][:], dtype=np.float32)
        n_f, n_a = freqs.shape[0], amps.shape[0]

        FF, AA = np.meshgrid(freqs, amps, indexing="ij")  # (n_f, n_a)

        # We'll accumulate per-simulation so we can hold one out cleanly
        X_per_sim = []
        y_per_sim = []
        per_sim_meta = []
        disp_per_sim = []
        sweep_per_sim = []

        sims_group = hdf5["simulations"]
        for sim_id in sims_group:
            sim_data = sims_group[sim_id]
            Q = float(sim_data.attrs["Q"])  # scalar
            gamma = float(sim_data.attrs["gamma"])  # scalar
            sweep_dir = encode_sweep_direction(sim_data.attrs["sweep_direction"])  # scalar

            # Read and reshape target grid
            disp = np.array(sim_data[:], dtype=np.float32)
            if disp.ndim == 1:
                disp = disp.reshape(n_f, n_a)
            elif disp.shape != (n_f, n_a):
                raise ValueError(
                    f"Simulation {sim_id}: unexpected displacement shape {disp.shape}, expected {(n_f, n_a)}"
                )

            # Build features for each grid point in this simulation
            Q_grid = np.full_like(FF, Q, dtype=np.float32)
            gamma_grid = np.full_like(FF, gamma, dtype=np.float32)
            sweep_grid = np.full_like(FF, sweep_dir, dtype=np.float32)

            X_local = np.stack([Q_grid, gamma_grid, sweep_grid, FF, AA], axis=-1).reshape(-1, 5)
            y_local = disp.reshape(-1, 1)

            X_per_sim.append(X_local)
            y_per_sim.append(y_local)
            per_sim_meta.append({"sim_id": sim_id, "Q": Q, "gamma": gamma, "sweep_direction": sweep_dir})
            disp_per_sim.append(disp)
            sweep_per_sim.append(float(sweep_dir))

        # Choose a holdout simulation (default: the middle one)
        holdout = None
        if HOLDOUT_ONE_SIM and len(per_sim_meta) > 0:
            if isinstance(HOLDOUT_SIM_INDEX, str) and HOLDOUT_SIM_INDEX.lower() == 'middle':
                hold_idx = len(per_sim_meta) // 2
            elif isinstance(HOLDOUT_SIM_INDEX, int):
                hold_idx = max(0, min(int(HOLDOUT_SIM_INDEX), len(per_sim_meta) - 1))
            else:
                hold_idx = len(per_sim_meta) // 2

            # Package holdout info
            hold_meta = per_sim_meta[hold_idx]
            hold_disp = disp_per_sim[hold_idx]
            hold_sd = sweep_per_sim[hold_idx]
            holdout = {
                "index": hold_idx,
                "meta": hold_meta,
                "disp": hold_disp,
                "sweep_dir": hold_sd,
            }

            # Use all other sims for training/validation
            keep_indices = [i for i in range(len(per_sim_meta)) if i != hold_idx]
        else:
            keep_indices = list(range(len(per_sim_meta)))

        X = np.concatenate([X_per_sim[i] for i in keep_indices], axis=0)
        y = np.concatenate([y_per_sim[i] for i in keep_indices], axis=0)

    # Normalize (z-score). Add small epsilon to avoid div-by-zero.
    eps = 1e-8
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + eps
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + eps

    norm = Normalizer(X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    Xn = norm.norm_X(X).astype(np.float32)
    yn = ((y - y_mean) / y_std).astype(np.float32)

    # Train/val split
    N = Xn.shape[0]
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(N)
    cut = int((1.0 - VALID_FRACTION) * N)
    idx_train, idx_val = perm[:cut], perm[cut:]
    X_train, y_train = Xn[idx_train], yn[idx_train]
    X_val, y_val = Xn[idx_val], yn[idx_val]

    aux = {"freqs": freqs, "amps": amps, "per_sim_meta": per_sim_meta, "holdout": holdout}
    return X_train, y_train, X_val, y_val, norm, aux

# ----------------------------
# Model
# ----------------------------
class MLP(nn.Module):
    hidden: Tuple[int, ...] = HIDDEN_SIZES

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.Dense(h)(x)
            x = nn.gelu(x)
        x = nn.Dense(1)(x)
        return x

# ----------------------------
# Training
# ----------------------------

def make_train_state(rng_key, model: MLP, input_dim: int) -> TrainState:
    params = model.init(rng_key, jnp.ones([1, input_dim], jnp.float32))['params']
    tx = optax.adam(LEARNING_RATE)
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def predict(params, x):
    return model.apply({'params': params}, x)

@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(params):
        y_pred = model.apply({'params': params}, x)
        loss = jnp.mean((y_pred - y) ** 2)
        return loss, y_pred
    (loss, y_pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.default_rng(RNG_SEED).shuffle(idx)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


def compute_metrics(params, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    y_pred = np.array(predict(params, jnp.array(X_val)))
    err = y_pred - y_val
    rmse = float(np.sqrt((err ** 2).mean()))
    mae = float(np.abs(err).mean())
    return {"rmse": rmse, "mae": mae}

# ----------------------------
# Inference helper: grid reconstruction
# ----------------------------

def predict_grid(params, norm: Normalizer, Q: float, gamma: float, sweep_direction: float,
                 freqs: np.ndarray, amps: np.ndarray) -> np.ndarray:
    """Predict a (n_f, n_a) grid of displacements for given parameters.

    Returns array with shape (n_f, n_a) in the ORIGINAL displacement units.
    """
    FF, AA = np.meshgrid(freqs.astype(np.float32), amps.astype(np.float32), indexing="ij")
    Qg = np.full_like(FF, float(Q), dtype=np.float32)
    Gg = np.full_like(FF, float(gamma), dtype=np.float32)
    Sd = np.full_like(FF, float(sweep_direction), dtype=np.float32)
    X = np.stack([Qg, Gg, Sd, FF, AA], axis=-1).reshape(-1, 5)
    Xn = ((X - norm.X_mean) / norm.X_std).astype(np.float32)

    y_pred_norm = np.array(predict(params, jnp.array(Xn)))  # (n_f*n_a, 1)
    y_pred = norm.denorm_y(y_pred_norm).reshape(FF.shape)
    return y_pred

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Load
    X_train, y_train, X_val, y_val, norm, aux = load_dataset(FILENAME)
    input_dim = X_train.shape[1]

    # Model/State
    global model
    model = MLP(HIDDEN_SIZES)
    rng = jax.random.PRNGKey(RNG_SEED)
    state = make_train_state(rng, model, input_dim)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        # iterate through mini-batches
        for xb, yb in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
            state, loss = train_step(state, jnp.array(xb), jnp.array(yb))

        if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            metrics = compute_metrics(state.params, X_val, y_val)
            print(f"Epoch {epoch:3d} | train_loss={float(loss):.5f} | val_rmse={metrics['rmse']:.5f} | val_mae={metrics['mae']:.5f}")

    # Example: predict a grid for the first simulation's meta
    meta0 = aux["per_sim_meta"][0]
    grid_pred = predict_grid(
        state.params,
        norm,
        Q=meta0["Q"],
        gamma=meta0["gamma"],
        sweep_direction=meta0["sweep_direction"],
        freqs=aux["freqs"],
        amps=aux["amps"],
    )
    print("Prediction grid shape:", grid_pred.shape)

    # Optional: save params & normalizer
    try:
        import pickle
        with open("duffing_mlp_params.pkl", "wb") as f:
            pickle.dump({"params": jax.device_get(state.params),
                         "norm": {
                             "X_mean": norm.X_mean,
                             "X_std": norm.X_std,
                             "y_mean": norm.y_mean,
                             "y_std": norm.y_std,
                         },
                         "config": {
                             "hidden_sizes": HIDDEN_SIZES,
                             "features": ["Q", "gamma", "sweep_direction", "freq", "amp"],
                         }}, f)
        print("Saved model to duffing_mlp_params.pkl")
    except Exception as e:
        print("Could not save params:", e)

# ----------------------------
# Parameter Estimation (Inverse)
# ----------------------------

def _norm_to_jnp(norm: Normalizer):
    return {
        "X_mean": jnp.array(norm.X_mean, dtype=jnp.float32),
        "X_std": jnp.array(norm.X_std, dtype=jnp.float32),
        "y_mean": jnp.array(norm.y_mean, dtype=jnp.float32),
        "y_std": jnp.array(norm.y_std, dtype=jnp.float32),
    }

@jax.jit
def predict_grid_jax(nn_params, norm_stats, Q, gamma, sweep_direction, freqs, amps):
    FF, AA = jnp.meshgrid(freqs, amps, indexing="ij")
    Qg = jnp.full_like(FF, Q)
    Gg = jnp.full_like(FF, gamma)
    Sd = jnp.full_like(FF, sweep_direction)
    X = jnp.stack([Qg, Gg, Sd, FF, AA], axis=-1).reshape(-1, 5)
    Xn = (X - norm_stats["X_mean"]) / norm_stats["X_std"]
    y_norm = model.apply({'params': nn_params}, Xn)
    y = y_norm * norm_stats["y_std"] + norm_stats["y_mean"]
    return y.reshape(FF.shape)

@jax.jit
def _mse_loss(nn_params, norm_stats, y_true, sweep_dir, freqs, amps, logQ, gamma):
    Q = jnp.exp(logQ)  # enforce Q>0
    y_pred = predict_grid_jax(nn_params, norm_stats, Q, gamma, sweep_dir, freqs, amps)
    return jnp.mean((y_pred - y_true) ** 2)


def estimate_Q_gamma_for_grid(nn_params,
                              norm: Normalizer,
                              y_true_grid: np.ndarray,
                              sweep_direction: float,
                              freqs: np.ndarray,
                              amps: np.ndarray,
                              init_Q: float = 100.0,
                              init_gamma: float = 0.0,
                              steps: int = 400,
                              lr: float = 5e-2,
                              verbose: bool = True):
    """Estimate (Q, gamma) by minimizing MSE between NN-predicted grid and a measured grid.

    Returns (Q_est, gamma_est, final_mse).
    """
    norm_stats = _norm_to_jnp(norm)
    y_true = jnp.array(y_true_grid, dtype=jnp.float32)
    f = jnp.array(freqs, dtype=jnp.float32)
    a = jnp.array(amps, dtype=jnp.float32)
    sd = jnp.array(float(sweep_direction), dtype=jnp.float32)

    # Optimize theta = (logQ, gamma)
    theta = jnp.array([jnp.log(jnp.array(init_Q, dtype=jnp.float32)), jnp.array(init_gamma, dtype=jnp.float32)])
    opt = optax.adam(lr)
    opt_state = opt.init(theta)

    @jax.jit
    def step(theta, opt_state):
        loss, grads = jax.value_and_grad(lambda th: _mse_loss(nn_params, norm_stats, y_true, sd, f, a, th[0], th[1]))(theta)
        updates, opt_state = opt.update(grads, opt_state, params=theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    best_loss = math.inf
    best_theta = None
    for i in range(steps):
        theta, opt_state, loss = step(theta, opt_state)
        if loss < best_loss:
            best_loss = float(loss)
            best_theta = np.array(theta)
        if verbose and (i % 50 == 0 or i == steps - 1):
            print(f"  [est] step {i:4d} loss={float(loss):.6e}")

    logQ_est, gamma_est = float(best_theta[0]), float(best_theta[1])
    Q_est = float(np.exp(logQ_est))
    return Q_est, gamma_est, float(best_loss)


def load_sim_grid(filename: str, sim_id: str, n_f: int, n_a: int):
    with h5py.File(filename, 'r') as hdf5:
        sim = hdf5['simulations'][sim_id]
        disp = np.array(sim[:], dtype=np.float32)
        if disp.ndim == 1:
            disp = disp.reshape(n_f, n_a)
        sweep_dir = encode_sweep_direction(sim.attrs['sweep_direction'])
    return disp, float(sweep_dir)


# ---- Parameter estimation demo on the HOLDOUT (middle) simulation ----
if __name__ == "__main__":
    if aux.get("holdout") is not None:
        hold = aux["holdout"]
        held_meta = hold["meta"]
        y_obs = hold["disp"]
        sweep_dir = hold["sweep_dir"]
        sim_id = held_meta["sim_id"]
        print(f"Estimating parameters for sim_id={sim_id} (true Q={held_meta['Q']:.6g}, true gamma={held_meta['gamma']:.6g}, sweep={sweep_dir})")

        Q_est, gamma_est, final_mse = estimate_Q_gamma_for_grid(
            state.params,
            norm,
            y_true_grid=y_obs,
            sweep_direction=sweep_dir,
            freqs=aux["freqs"],
            amps=aux["amps"],
            init_Q=max(1.0, held_meta["Q"] * 0.7),  # a rough guess
            init_gamma=held_meta["gamma"] * 0.5,
            steps=500,
            lr=5e-2,
            verbose=True,
        )
        print(f"Estimated Q ≈ {Q_est:.6g}, gamma ≈ {gamma_est:.6g} | True Q={held_meta['Q']:.6g}, True gamma={held_meta['gamma']:.6g}, MSE={final_mse:.6e}")
    else:
        print("No holdout simulation configured; set HOLDOUT_ONE_SIM=True to enable middle-sim estimation.")
