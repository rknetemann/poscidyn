from jaxtyping import Array, Float, PyTree
import jax.numpy as jnp
import jax
from typing import Any
import equinox as eqx
from pathlib import Path

import oscidynn

from .utils import linear_response_amplitudes

# Dataset
DATAFILES = Path(
    "/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/converted"
)

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.eqx"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.eqx"
STATE_PATH = CHECKPOINT_DIR / "train_state.json"
NORMALIZER_PATH = CHECKPOINT_DIR / "normalizer.eqx"

# History
HISTORY_JSONL_PATH = CHECKPOINT_DIR / "history.jsonl"
HISTORY_CSV_PATH = CHECKPOINT_DIR / "history.csv"

# Regularization parameters
Q_LOSS_REG = 1.0
OMEGA_0_LOSS_REG = 1.0
ALPHA_LOSS_REG = 1.0
GAMMA_LOSS_REG = 1.0

@eqx.filter_jit
def loss(
    model: oscidynn.MultiLayerPerceptron,
    normalizer: oscidynn.DatasetNormalizer,
    x: PyTree[Float[Array, "batch ..."]],
    y: PyTree[Float[Array, "batch ..."]],
    kwargs: PyTree[Any, "..."],
) -> Float[Array, ""]:
    
    def loss_per_sample(x_i, y_i, kwargs_i):
        x_i_norm = normalizer.norm_X(x_i)
        y_i_norm = normalizer.norm_Y(y_i)

        amps_linear = linear_response_amplitudes(y_i, kwargs_i)
        difference = x_i['amplitudes'] - amps_linear
        difference_rel_l2 = jnp.linalg.norm(difference) / (jnp.linalg.norm(x_i['amplitudes']) + 1e-12)
        # --- Map rel_l2 -> weight in [0, 1] ---
        threshold = 0.10
        sharpness = 50.0 
        w_nonlinear_terms = jax.nn.sigmoid(sharpness * (difference_rel_l2 - threshold)) 
        
        pred_y_norm = model(x_i_norm)

        Q_loss = jnp.mean((pred_y_norm['Q'] - y_i_norm['Q'])**2) * Q_LOSS_REG
        omega_0_loss = jnp.mean((pred_y_norm['omega_0'] - y_i_norm['omega_0'])**2) * OMEGA_0_LOSS_REG
        alpha_loss = jnp.mean((pred_y_norm['alpha'] - y_i_norm['alpha'])**2) * ALPHA_LOSS_REG
        gamma_loss = jnp.mean((pred_y_norm['gamma'] - y_i_norm['gamma'])**2) * GAMMA_LOSS_REG * w_nonlinear_terms

        return jnp.mean(Q_loss + omega_0_loss + alpha_loss + gamma_loss)

    loss = jnp.mean(jax.vmap(loss_per_sample)(x, y, kwargs))
    
    return loss

def evaluate():
    pass

def train():
    pass

if __name__ == "__main__":
    pass