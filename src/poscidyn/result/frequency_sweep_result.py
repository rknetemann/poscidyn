from dataclasses import dataclass
from typing import Any

from jax import tree_util
from jaxtyping import Array, PyTree

@dataclass
class Phasors():
    amplitudes: PyTree[Array]
    phases: PyTree[Array] | None
    demod_freqs: PyTree[Array] | None

@tree_util.register_pytree_node_class
@dataclass
class FrequencySweep:
    modal_coordinates: Phasors
    modal_superposition: Phasors
    stats: dict[str, Any]

    def tree_flatten(self):
        leaves = (
            self.modal_coordinates.amplitudes,
            self.modal_coordinates.phases,
            self.modal_coordinates.demod_freqs,
            self.modal_superposition.amplitudes,
            self.modal_superposition.phases,
            self.modal_superposition.demod_freqs,
            self.stats,
        )
        return leaves, None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        (
            modal_coord_amps,
            modal_coord_phases,
            modal_coord_demod,
            modal_super_amps,
            modal_super_phases,
            modal_super_demod,
            stats,
        ) = leaves
        return cls(
            modal_coordinates=Phasors(
                amplitudes=modal_coord_amps,
                phases=modal_coord_phases,
                demod_freqs=modal_coord_demod,
            ),
            modal_superposition=Phasors(
                amplitudes=modal_super_amps,
                phases=modal_super_phases,
                demod_freqs=modal_super_demod,
            ),
            stats=stats,
        )
