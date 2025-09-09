import os
from typing import Optional, Tuple

import optax
import equinox as eqx
import jax
import jax.numpy as jnp

class Trainer():
    def __init__(self, model: eqx.Module, optimizer: optax.GradientTransformation, seed: int = 0, use_key: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.use_key = use_key

        self.opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))
        
        self.rng = jax.random.PRNGKey(seed)
        
        self._train_step = self._build_train_step()
        
    def _build_train_step(self):
        use_key = self.use_key
        optimizer = self.optimizer

        @eqx.filter_jit
        def train_step(model: eqx.Module, opt_state, x: jnp.ndarray, y: jnp.ndarray, key):
            def loss_fn(m: eqx.Module):
                preds = model(x, key=key) if use_key else model(x)
                loss = jnp.mean((preds - y) ** 2)
                return loss

            loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
            updates, opt_state = optimizer.update(
                grads, opt_state, params=eqx.filter(model, eqx.is_inexact_array)
            )
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        return train_step
    
    def _build_eval_step(self):
        use_key = self.use_key

        @eqx.filter_jit
        def eval_step(model: eqx.Module, x: jnp.ndarray, y: jnp.ndarray, key):
            logits = model(x, key=key) if use_key else model(x)
            loss = Trainer._ce_with_integer_labels(logits, y)
            return loss

        return eval_step        
    
    def _run_epoch(self, loader, train: bool) -> Tuple[float, float]:
        total_loss = 0.0
        total_n = 0
        
        for x, y in loader:
            bsz = int(x.shape[0])
            total_n += bsz

            self.rng, key = jax.random.split(self.rng)

            if train:
                self.model, self.opt_state, loss = self._train_step(
                    self.model, self.opt_state, x, y, key
                )
            else:
                loss = self._eval_step(self.model, x, y, key)

            total_loss += float(loss) * bsz

        return total_loss / total_n
    

    def train(self, train_loader, val_loader, 
              epochs: int, verbose: bool = True, 
              early_stop_patience: Optional[int] = 5, checkpoint_path: Optional[str] = None):
        
        best_val = float('inf')
        patience = 0
        
        for epoch in range(1, epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            
            if verbose:
                print(f"Epoch {epoch}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} - "
                      f"Val Loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                if checkpoint_path:
                    self.save_model(checkpoint_path)
            else:
                patience += 1
                if early_stop_patience is not None and patience >= early_stop_patience:
                    if verbose:
                        print("Early stopping.")
                    break

    def test(self, test_loader, verbose: bool = True):
        test_loss = self._run_epoch(test_loader, train=False)
        
        if verbose:
            print(f"Test loss {test_loss:.4f}")

        return test_loss

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        eqx.tree_serialise_leaves(path, self.model)

    def load_model(self, path: str):
        self.model = eqx.tree_deserialise_leaves(path, self.model)