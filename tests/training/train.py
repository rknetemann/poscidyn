from pathlib import Path
import numpy as np
import oscidynn
import jax
from jaxtyping import Array, Float, PyTree, Int
import jax.numpy as jnp
import equinox as eqx
import optax

DATAFILES = Path('/home/raymo/Projects/oscidyn/data/simulations/')

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678

key = jax.random.PRNGKey(SEED)

dataloader = oscidynn.DataLoader(DATAFILES)

pairs = dataloader.load_data(0, [0])

key, subkey = jax.random.split(key, 2)
model = oscidynn.MultiLayerPerceptron(x_shape=pairs[0][0].size, y_shape=pairs[0][1].size, key=subkey)

def _batch_from_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[jnp.ndarray, jnp.ndarray]:
    xs, ys = zip(*pairs)
    x_batch = jnp.stack([jnp.asarray(x) for x in xs])
    y_batch = jnp.stack([jnp.asarray(y) for y in ys])
    return x_batch, y_batch

def loss(
    model: oscidynn.MultiLayerPerceptron, x: Float[Array, "batch input_dim"], y: Float[Array, "batch output_dim"]
) -> Float[Array, ""]:
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=0)
    if y.ndim == 1:
        y = jnp.expand_dims(y, axis=0)
    pred_y = jax.vmap(model)(x)
    return mean_squared_error(y, pred_y)

def mean_squared_error(
    y: Float[Array, "batch output_dim"], pred_y: Float[Array, "batch output_dim"]
) -> Float[Array, ""]:
    return jnp.mean((pred_y - y) ** 2)

loss_value = loss(model, pairs[0][0], pairs[0][1])
example_input = pairs[0][0]
if example_input.ndim == 1:
    example_input = jnp.expand_dims(example_input, axis=0)
output = jax.vmap(model)(example_input)

value, grads = eqx.filter_value_and_grad(loss)(model, pairs[0][0], pairs[0][1])

loss = eqx.filter_jit(loss)

@eqx.filter_jit
def compute_accuracy(
    model: oscidynn.MultiLayerPerceptron, x: Float[Array, "batch input_dim"], y: Float[Array, "batch output_dim"]
) -> Float[Array, ""]:
    if x.ndim == 1:
        x = jnp.expand_dims(x, axis=0)
    if y.ndim == 1:
        y = jnp.expand_dims(y, axis=0)
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)

test_data = dataloader.load_data(1, 'all')

@eqx.filter_jit
def evaluate(model: oscidynn.MultiLayerPerceptron):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0

    for (x, y) in test_data:
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        avg_loss += loss(model, x, y)
        avg_acc += compute_accuracy(model, x, y)
    return avg_loss / len(test_data), avg_acc / len(test_data)


optim = optax.adamw(LEARNING_RATE)

def train(
    model: oscidynn.MultiLayerPerceptron,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> oscidynn.MultiLayerPerceptron:

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    rng = np.random.default_rng(SEED)
    n_train = dataloader.n_split_sims[0]

    @eqx.filter_jit
    def make_step(
        model: oscidynn.MultiLayerPerceptron,
        opt_state: PyTree,
        x: Float[Array, "batch input_dim"],
        y: Float[Array, "batch output_dim"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def batch_index_iterator(batch_size: int):
        """Yield shuffled batches of split indices indefinitely for streaming loads."""
        indices = np.arange(n_train)
        while True:
            rng.shuffle(indices)
            for start in range(0, n_train, batch_size):
                yield indices[start:start + batch_size]

    batch_iter = batch_index_iterator(BATCH_SIZE)

    print("Starting training...")
    for step in range(steps):
        batch_indices = next(batch_iter)
        pairs = dataloader.load_data(0, batch_indices.tolist())
        x_batch, y_batch = _batch_from_pairs(pairs)
        model, opt_state, train_loss = make_step(model, opt_state, x_batch, y_batch)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model

try:
    model = train(model, optim, STEPS, PRINT_EVERY)
finally:
    dataloader.close()
