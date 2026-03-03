from .abstract_sweep import AbstractSweep
from .sweep_directions import AbstractSweepDirection
from .sweep_directions import Forward, Backward
import jax.numpy as jnp
from jax import lax

class NearestNeighbourSweep(AbstractSweep):
    def __init__(self, sweep_direction: AbstractSweepDirection | list[AbstractSweepDirection] = [Forward(), Backward()]):
        super().__init__()
        self.sweep_direction = sweep_direction

    def to_dtype(self, dtype):
        return NearestNeighbourSweep(sweep_direction=self.sweep_direction)

    def sweep(self, periodic_solutions):
        """
        Expects:
          periodic_solutions['amplitude'] with shape
          (n_freq, n_amp, n_init_disp, n_init_vel)
          periodic_solutions['phase'] with the same shape.

        Selection is performed in amplitude space. The chosen phase is taken
        from the same seed index as the chosen amplitude.

        Returns:
          forward/backward: amplitude arrays with shape (n_freq, n_amp)
          forward_phase/backward_phase: corresponding phase arrays
          forward_idx/backward_idx: flat chosen seed indices (debug)
        """
        amplitudes = periodic_solutions['amplitude']
        phases = periodic_solutions['phase']
        
        n_freq, n_amp, _, _ = amplitudes.shape

        results = {
            'forward': None,
            'backward': None,
            'forward_phase': None,
            'backward_phase': None,
            'forward_idx': None,
            'backward_idx': None,
        }

        for direction_obj in (self.sweep_direction if isinstance(self.sweep_direction, list) else [self.sweep_direction]):
            if isinstance(direction_obj, Forward):
                order = jnp.arange(n_freq)
                start_idx = 0
                direction = 'forward'
            elif isinstance(direction_obj, Backward):
                order = jnp.arange(n_freq - 1, -1, -1)
                start_idx = n_freq - 1
                direction = 'backward'
            else:
                raise ValueError("Unsupported sweep direction")

            def sweep_one_amplitude(amp_idx):
                # candidates at the starting frequency, flattened over all seeds
                start_cands = amplitudes[start_idx, amp_idx].reshape(-1)
                start_phase_cands = phases[start_idx, amp_idx].reshape(-1)

                # choose a reasonable start: the largest finite response (or first finite)
                finite_mask = jnp.isfinite(start_cands)
                # if all NaN/inf, default to zero
                start_choice = jnp.where(
                    finite_mask.any(),
                    start_cands[jnp.argmax(jnp.where(finite_mask, start_cands, -jnp.inf))],
                    0.0,
                )
                start_choice_idx = jnp.where(
                    finite_mask.any(),
                    jnp.argmax(jnp.where(finite_mask, start_cands, -jnp.inf)),
                    -1,
                )
                safe_start_idx = jnp.maximum(start_choice_idx, 0)
                start_choice_phase = jnp.where(
                    finite_mask.any(),
                    start_phase_cands[safe_start_idx],
                    jnp.nan,
                )

                def body(carry, freq_idx):
                    prev_val, prev_phase, out_vals, out_phases, out_idx = carry
                    cands = amplitudes[freq_idx, amp_idx].reshape(-1)  # (n_seeds,)
                    phase_cands = phases[freq_idx, amp_idx].reshape(-1)  # (n_seeds,)
                    # distance in *amplitude* space
                    diffs = jnp.abs(cands - prev_val)

                    # ignore NaNs/Infs by giving them infinite distance
                    finite_mask = jnp.isfinite(cands)
                    diffs = jnp.where(finite_mask, diffs, jnp.inf)

                    k = jnp.argmin(diffs)
                    no_valid = jnp.logical_not(jnp.any(finite_mask))
                    chosen = jnp.where(no_valid, prev_val, cands[k])
                    chosen_phase = jnp.where(no_valid, prev_phase, phase_cands[k])
                    chosen_idx = jnp.where(no_valid, -1, k)

                    out_vals = out_vals.at[freq_idx].set(chosen)
                    out_phases = out_phases.at[freq_idx].set(chosen_phase)
                    out_idx = out_idx.at[freq_idx].set(chosen_idx)
                    return (chosen, chosen_phase, out_vals, out_phases, out_idx), None

                init_vals = jnp.full((n_freq,), jnp.nan)
                init_phases = jnp.full((n_freq,), jnp.nan)
                init_idx = jnp.full((n_freq,), -1, dtype=jnp.int32)

                (_, _, out_vals, out_phases, out_idx), _ = lax.scan(
                    body, (start_choice, start_choice_phase, init_vals, init_phases, init_idx), order
                )
                return out_vals, out_phases, out_idx

            vals, phase_vals, idxs = zip(*(sweep_one_amplitude(amp) for amp in range(n_amp)))
            sweeped_vals = jnp.stack(vals, axis=1)  # (n_freq, n_amp)
            sweeped_phase_vals = jnp.stack(phase_vals, axis=1)  # (n_freq, n_amp)
            sweeped_phase_vals = jnp.unwrap(sweeped_phase_vals, axis=0)
            sweeped_idxs = jnp.stack(idxs, axis=1)  # (n_freq, n_amp)
            results[direction] = sweeped_vals
            results[f"{direction}_phase"] = sweeped_phase_vals
            results[f"{direction}_idx"] = sweeped_idxs

        return results
