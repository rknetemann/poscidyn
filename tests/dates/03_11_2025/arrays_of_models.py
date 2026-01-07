import numpy as np
import poscidyn
import jax

from typing import Dict, Any, List, Tuple, NamedTuple, Union, Callable, Optional, Type
from jaxtyping import PyTree

Q, omega_0, alpha, gamma = np.array([50.0, 23.0, 23.0]), np.array([1.0, 2.0, 3.0]), np.zeros((3,3,3)), np.zeros((3,3,3,3))
model1 = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)

Q, omega_0, alpha, gamma = np.array([20.0, 253.0, 23.0]), np.array([1.0, 2.0, 3.0]), np.zeros((3,3,3)), np.zeros((3,3,3,3))
model2 = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)


test = dict(
    model1 = model1,
    model2 = model2,
)
flattened, _ = jax.tree_util.tree_flatten_with_path(test)

for key_path, value in flattened:
  print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')
  
print(type(test) == PyTree)
print(type(test))