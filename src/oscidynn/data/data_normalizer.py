import equinox as eqx
import jax.numpy as jnp

class DatasetNormalizer(eqx.Module):
    x_mean: jnp.ndarray
    x_std: jnp.ndarray
    y_mean: jnp.ndarray
    y_std: jnp.ndarray

    @classmethod
    def from_data(cls, X: jnp.ndarray, Y: jnp.ndarray, eps: float = 1e-8) -> "DatasetNormalizer":
        x_mean = jnp.asarray(X.mean(axis=0, keepdims=True))
        x_std = jnp.asarray(X.std(axis=0, keepdims=True) + eps)
        y_mean = jnp.asarray(Y.mean(axis=0, keepdims=True))
        y_std = jnp.asarray(Y.std(axis=0, keepdims=True) + eps)
        return cls(x_mean, x_std, y_mean, y_std)

    def norm_X(self, X: jnp.ndarray | jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(X) - self.x_mean) / self.x_std

    def norm_Y(self, Y: jnp.ndarray | jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(Y) - self.y_mean) / self.y_std

    def denorm_Y(self, Y: jnp.ndarray) -> jnp.ndarray:
        return Y * self.y_std + self.y_mean