import jax.numpy as jnp

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