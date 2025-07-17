# models.py
from __future__ import annotations
from dataclasses import dataclass, field
import jax

oscimodel = lambda cls: dataclass(eq=False)(cls)

@oscimodel
class AbstractModel:
    rhs_jit: callable = field(init=False, repr=False)

    n_modes: int = field(init=True, repr=False)

    def __post_init__(self):
        self._build_rhs()

    def _build_rhs(self):
        # ensure the subclass has implemented an `rhs` method
        if not hasattr(self, "rhs") or not callable(getattr(self, "rhs")):
            raise NotImplementedError(f"{type(self).__name__}.rhs() is not implemented.")
        # JIT‐compile the RHS
        self.rhs_jit = jax.jit(self.rhs)

