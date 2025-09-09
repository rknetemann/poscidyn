import os; os.environ['JAX_PLATFORM_NAME'] = 'cpu'
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree 
import h5py
import optax

FILENAME = Path("/home/raymo/Downloads/batch_1.hdf5")
INPUT_SHAPE = (200,)
SEED = 42
LEARNING_RATE = 3e-4

class MLP(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4, key5, key6 = jax.random.split(key, 6)

        self.layers = [
            eqx.nn.Linear(200, 128, key=key4),
            jax.nn.relu,
            eqx.nn.Linear(128, 64, key=key5),
            jax.nn.relu,
            eqx.nn.Linear(64, 2, key=key6),
        ]

    def __call__(self, x: Float[Array, "200"]) -> Float[Array, "2"]:
        for layer in self.layers:
            x = layer(x)
        return x
    
def loss(model: MLP, x: Float[Array, "batch 200"], y: Float[Array, "batch 2"]) -> Float[Array, ""]:
    preds = jax.vmap(model)(x)  # (batch, 2)
    return jnp.mean(jnp.sum((preds - y) ** 2, axis=-1))  # MSE over both outputs

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)
model = MLP(subkey)


test_x = jax.random.normal(key, (28, 200))     # (batch=28, 200)
test_y = jax.random.normal(key, (28, 2))       # (batch=28, 2)
pred_y = jax.vmap(model)(test_x)               # (28, 2)
print("Predicited y: ", pred_y[0])                               # show first prediction
value, grads = eqx.filter_value_and_grad(loss)(model, test_x, test_y)
print("Loss value: ", value)


loss = eqx.filter_jit(loss)  # JIT our loss function from earlier!

@eqx.filter_jit
def compute_accuracy(
    model: MLP, x: Float[Array, "batch 200"], y: Int[Array, "batch 2"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    accuracy = jnp.mean((pred_y == y).astype(jnp.float32))
    return accuracy

def evaluate(model: MLP, test_dataset):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    for x, y in test_dataset:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(test_dataset), avg_acc / len(test_dataset)


with h5py.File(FILENAME, "r") as f:
    freqs = jnp.array(f["driving_frequencies"][:], dtype=float)
    amps  = jnp.array(f["driving_amplitudes"][:], dtype=float)
    n_f, n_a = freqs.shape[0], amps.shape[0]

    total_disp_amps = f["simulations"]

    for name, obj in total_disp_amps.items():
        if isinstance(obj, h5py.Dataset):
            data = obj[...]
            
            
optim = optax.adamw(LEARNING_RATE)

def train(
    model: MLP,
    train_dataset,
    test_dataset,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> MLP:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: MLP,
        opt_state: PyTree,
        x: Float[Array, "batch 200"],
        y: Int[Array, "batch 2"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model