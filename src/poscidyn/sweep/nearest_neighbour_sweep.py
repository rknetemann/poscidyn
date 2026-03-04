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
        amplitudes = periodic_solutions['amplitude']
        phases = periodic_solutions['phase']
        if amplitudes.ndim < 4:
            raise ValueError(
                "periodic_solutions['amplitude'] must have at least 4 dimensions: "
                "(n_freq, n_amp, n_init_disp, n_init_vel, ...response_dims)"
            )
        if phases.shape != amplitudes.shape:
            raise ValueError("periodic_solutions['phase'] must have the same shape as amplitude")

        n_freq, n_amp, n_init_disp, n_init_vel = amplitudes.shape[:4]
        response_shape = amplitudes.shape[4:]
        n_seeds = n_init_disp * n_init_vel

        amplitudes_seed = amplitudes.reshape((n_freq, n_amp, n_seeds) + response_shape)
        phases_seed = phases.reshape((n_freq, n_amp, n_seeds) + response_shape)

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
                # candidates at the starting frequency, flattened over seeds only
                start_cands = amplitudes_seed[start_idx, amp_idx]  # (n_seeds, *response_shape)
                start_phase_cands = phases_seed[start_idx, amp_idx]  # (n_seeds, *response_shape)
                start_cands_flat = start_cands.reshape((n_seeds, -1))

                # choose the largest finite RMS amplitude as start
                finite_mask = jnp.all(jnp.isfinite(start_cands_flat), axis=1)
                start_rms = jnp.sqrt(jnp.mean(start_cands_flat**2, axis=1))
                safe_start_rms = jnp.where(finite_mask, start_rms, -jnp.inf)
                start_choice_idx_raw = jnp.argmax(safe_start_rms)
                has_valid_start = jnp.any(finite_mask)
                start_choice = jnp.where(
                    has_valid_start,
                    start_cands[start_choice_idx_raw],
                    jnp.zeros(response_shape, dtype=amplitudes.dtype),
                )
                start_choice_idx = jnp.where(
                    has_valid_start,
                    start_choice_idx_raw,
                    -1,
                )
                safe_start_idx = jnp.maximum(start_choice_idx, 0)
                start_choice_phase = jnp.where(
                    has_valid_start,
                    start_phase_cands[safe_start_idx],
                    jnp.full(response_shape, jnp.nan, dtype=phases.dtype),
                )

                def body(carry, freq_idx):
                    prev_val, prev_phase, out_vals, out_phases, out_idx = carry
                    cands = amplitudes_seed[freq_idx, amp_idx]  # (n_seeds, *response_shape)
                    phase_cands = phases_seed[freq_idx, amp_idx]  # (n_seeds, *response_shape)
                    cands_flat = cands.reshape((n_seeds, -1))
                    prev_flat = prev_val.reshape((1, -1))

                    # RMS distance over all response components (modes/multiples)
                    diffs_flat = cands_flat - prev_flat
                    diffs = jnp.sqrt(jnp.mean(diffs_flat**2, axis=1))  # (n_seeds,)

                    # ignore NaN/Inf candidates
                    finite_mask = jnp.all(jnp.isfinite(cands_flat), axis=1)
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

                init_vals = jnp.full((n_freq,) + response_shape, jnp.nan, dtype=amplitudes.dtype)
                init_phases = jnp.full((n_freq,) + response_shape, jnp.nan, dtype=phases.dtype)
                init_idx = jnp.full((n_freq,), -1, dtype=jnp.int32)

                (_, _, out_vals, out_phases, out_idx), _ = lax.scan(
                    body, (start_choice, start_choice_phase, init_vals, init_phases, init_idx), order
                )
                return out_vals, out_phases, out_idx

            vals, phase_vals, idxs = zip(*(sweep_one_amplitude(amp) for amp in range(n_amp)))
            sweeped_vals = jnp.stack(vals, axis=1)  # (n_freq, n_amp, *response_shape)
            sweeped_phase_vals = jnp.stack(phase_vals, axis=1)  # (n_freq, n_amp, *response_shape)
            finite_phase_mask = jnp.isfinite(sweeped_phase_vals)
            max_abs_phase = jnp.max(jnp.abs(jnp.where(finite_phase_mask, sweeped_phase_vals, 0.0)))
            has_finite_phase = jnp.any(finite_phase_mask)
            should_unwrap = jnp.logical_and(has_finite_phase, max_abs_phase <= (jnp.pi + 1e-6))
            sweeped_phase_vals = jnp.where(
                should_unwrap,
                jnp.unwrap(sweeped_phase_vals, axis=0),
                sweeped_phase_vals,
            )
            sweeped_idxs = jnp.stack(idxs, axis=1)  # (n_freq, n_amp)
            results[direction] = sweeped_vals
            results[f"{direction}_phase"] = sweeped_phase_vals
            results[f"{direction}_idx"] = sweeped_idxs

        return results
