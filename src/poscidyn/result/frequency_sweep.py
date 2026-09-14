from dataclasses import dataclass
from typing import Any

from jax import tree_util
from jaxtyping import Array, PyTree

@tree_util.register_pytree_node_class
@dataclass
class ResponseData:
    value: PyTree[Array]

    @property
    def amplitude(self):
        return self.value

    @property
    def amplitudes(self):
        return self.value

    @property
    def phases(self):
        return None

    @property
    def demod_freqs(self):
        return None

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class DemodulationResult(ResponseData):
    phase: PyTree[Array] | None = None
    response_frequency: PyTree[Array] | None = None

    def tree_flatten(self):
        return (self.value, self.phase, self.response_frequency), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
@dataclass
class ScalarResponseResult(ResponseData):
    measure: str = "value"

    def tree_flatten(self):
        return (self.value,), self.measure

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data)

    @property
    def name(self):
        return self.measure

    @property
    def rms(self):
        if self.measure != "rms":
            raise AttributeError("This result was not produced by RMS")
        return self.value

    @property
    def minimum(self):
        if self.measure != "minimum":
            raise AttributeError("This result was not produced by Min")
        return self.value

    @property
    def maximum(self):
        if self.measure != "maximum":
            raise AttributeError("This result was not produced by Max")
        return self.value


Phasors = DemodulationResult


@tree_util.register_pytree_node_class
@dataclass
class BranchResult:
    modal: ResponseData
    total: ResponseData

    @property
    def successful(self):
        return None

    def tree_flatten(self):
        return (self.modal, self.total), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@tree_util.register_pytree_node_class
@dataclass
class FrequencySweep:
    frequency: Array
    forward: BranchResult
    backward: BranchResult
    stats: dict[str, Any]

    def tree_flatten(self):
        leaves = (
            self.frequency,
            self.forward,
            self.backward,
            self.stats,
        )
        return leaves, None

    @classmethod
    def tree_unflatten(cls, aux_data, leaves):
        (
            frequency,
            forward,
            backward,
            stats,
        ) = leaves
        return cls(
            frequency=frequency, forward=forward, backward=backward,
            stats=stats,
        )

    # Compatibility aliases for the pre-BranchResult API.
    @property
    def modal_coordinates(self):
        return ResponseData(
            amplitude={"forward": self.forward.modal.amplitude, "backward": self.backward.modal.amplitude},
            phase={"forward": self.forward.modal.phase, "backward": self.backward.modal.phase},
            response_frequency={"forward": self.forward.modal.response_frequency, "backward": self.backward.modal.response_frequency},
        )

    @property
    def modal_superposition(self):
        return ResponseData(
            amplitude={"forward": self.forward.total.amplitude, "backward": self.backward.total.amplitude},
            phase={"forward": self.forward.total.phase, "backward": self.backward.total.phase},
            response_frequency={"forward": self.forward.total.response_frequency, "backward": self.backward.total.response_frequency},
        )
