from __future__ import annotations
from dataclasses import field, dataclass

oscimodel = lambda cls: dataclass(eq=False, kw_only=True)(cls)

@oscimodel
class AbstractModel:
    rhs_jit: callable = field(init=False, repr=False)

    def __post_init__(self):
        pass
    