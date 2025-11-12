import jax.numpy as jnp
from jax import tree_util

class OneToneExcitation:
    def __init__(self, drive_frequencies, drive_amplitudes, modal_forces):
        self.drive_frequencies = drive_frequencies
        self.drive_amplitudes = drive_amplitudes
        self.modal_forces = modal_forces
        
        self.f_omegas = jnp.asarray(drive_frequencies)
        self.f_amps = jnp.outer(jnp.asarray(drive_amplitudes), jnp.asarray(modal_forces))
        
    def to_dtype(self, dtype):
        return OneToneExcitation(
            drive_frequencies=self.drive_frequencies.astype(dtype),
            drive_amplitudes=self.drive_amplitudes.astype(dtype),
            modal_forces=self.modal_forces.astype(dtype)
        )

# Register OneToneExcitation as a pytree
def _tree_flatten(obj):
    leaves = (obj.f_omegas, obj.f_amps)
    aux_data = (obj.drive_frequencies, obj.drive_amplitudes, obj.modal_forces)
    return leaves, aux_data

def _tree_unflatten(aux_data, leaves):
    f_omegas, f_amps = leaves
    drive_frequencies, drive_amplitudes, modal_forces = aux_data
    obj = OneToneExcitation(drive_frequencies, drive_amplitudes, modal_forces)
    obj.f_omegas = f_omegas
    obj.f_amps = f_amps
    return obj

tree_util.register_pytree_node(OneToneExcitation, _tree_flatten, _tree_unflatten)