from .abstract_sweep import AbstractSweep
from .sweep_directions import AbstractSweepDirection
from .sweep_directions import Forward, Backward
import jax.numpy as jnp
from equinox import filter_jit
from jax import lax

class NearestNeighbourSweep(AbstractSweep):
    def __init__(
        self,
        sweep_direction: AbstractSweepDirection | list[AbstractSweepDirection] = [Forward(), Backward()],
        phase_weight: float = 0.25,
        seed_switch_penalty: float = 0.05,
    ):
        super().__init__()
        self.sweep_direction = sweep_direction
        self.phase_weight = phase_weight
        self.seed_switch_penalty = seed_switch_penalty

    def to_dtype(self, dtype):
        return NearestNeighbourSweep(
            sweep_direction=self.sweep_direction,
            phase_weight=self.phase_weight,
            seed_switch_penalty=self.seed_switch_penalty,
        )

    @filter_jit
    def sweep(self, periodic_solutions):
        amplitudes = periodic_solutions['amplitude']
        phases = periodic_solutions['phase']

        n_freq, n_amp, n_init_disp, n_init_vel = amplitudes.shape[:4]
        response_shape = amplitudes.shape[4:]
        n_seeds = n_init_disp * n_init_vel

        # we reshape the amplitudes to collapse the initial conditions into one
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

        # sweep_direction can be either a list of the directions or a single direction
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
                    prev_val, prev_phase, prev_idx, out_vals, out_phases, out_idx = carry
                    cands = amplitudes_seed[freq_idx, amp_idx]  # (n_seeds, *response_shape)
                    phase_cands = phases_seed[freq_idx, amp_idx]  # (n_seeds, *response_shape)
                    cands_flat = cands.reshape((n_seeds, -1))
                    prev_flat = prev_val.reshape((1, -1))
                    phase_cands_flat = phase_cands.reshape((n_seeds, -1))
                    prev_phase_flat = prev_phase.reshape((1, -1))

                    # RMS amplitude distance over all response components.
                    amp_diffs_flat = cands_flat - prev_flat
                    amp_diffs = jnp.sqrt(jnp.mean(amp_diffs_flat**2, axis=1))  # (n_seeds,)
                    prev_amp_scale = jnp.sqrt(jnp.mean(prev_flat**2)) + 1e-12
                    amp_cost = amp_diffs / prev_amp_scale

                    # Circular phase distance; ignored where phase is not finite.
                    phase_delta = jnp.angle(jnp.exp(1j * (phase_cands_flat - prev_phase_flat)))
                    phase_finite_mask = jnp.logical_and(
                        jnp.isfinite(phase_cands_flat),
                        jnp.isfinite(prev_phase_flat),
                    )
                    phase_count = jnp.sum(phase_finite_mask, axis=1)
                    safe_phase_count = jnp.maximum(phase_count, 1)
                    phase_mse = jnp.sum(
                        jnp.where(phase_finite_mask, phase_delta**2, 0.0),
                        axis=1,
                    ) / safe_phase_count
                    phase_rms = jnp.sqrt(phase_mse)
                    phase_cost = phase_rms / jnp.pi
                    has_phase_info = phase_count > 0

                    diffs = amp_cost + self.phase_weight * jnp.where(has_phase_info, phase_cost, 0.0)
                    switch_cost = jnp.where(
                        prev_idx >= 0,
                        jnp.where(jnp.arange(n_seeds) != prev_idx, self.seed_switch_penalty, 0.0),
                        0.0,
                    )
                    diffs = diffs + switch_cost

                    # ignore NaN/Inf candidates
                    finite_mask = jnp.all(jnp.isfinite(cands_flat), axis=1)
                    diffs = jnp.where(finite_mask, diffs, jnp.inf)

                    k = jnp.argmin(diffs)
                    no_valid = jnp.logical_not(jnp.any(finite_mask))
                    chosen = jnp.where(no_valid, prev_val, cands[k])
                    chosen_phase = jnp.where(no_valid, prev_phase, phase_cands[k])
                    chosen_idx = jnp.where(no_valid, prev_idx, k)

                    out_vals = out_vals.at[freq_idx].set(chosen)
                    out_phases = out_phases.at[freq_idx].set(chosen_phase)
                    out_idx = out_idx.at[freq_idx].set(chosen_idx)
                    return (chosen, chosen_phase, chosen_idx, out_vals, out_phases, out_idx), None

                init_vals = jnp.full((n_freq,) + response_shape, jnp.nan, dtype=amplitudes.dtype)
                init_phases = jnp.full((n_freq,) + response_shape, jnp.nan, dtype=phases.dtype)
                init_idx = jnp.full((n_freq,), -1, dtype=jnp.int32)

                (_, _, _, out_vals, out_phases, out_idx), _ = lax.scan(
                    body, (start_choice, start_choice_phase, start_choice_idx, init_vals, init_phases, init_idx), order
                )
                return out_vals, out_phases, out_idx

            vals, phase_vals, idxs = zip(*(sweep_one_amplitude(amp) for amp in range(n_amp)))
            sweeped_vals = jnp.stack(vals, axis=1)  # (n_freq, n_amp, *response_shape)
            sweeped_phase_vals = jnp.stack(phase_vals, axis=1)  # (n_freq, n_amp, *response_shape)

            # Backward compatibility for single-mode/single-multiple responses:
            # drop singleton response dimensions so plotting code that expects
            # (n_freq, n_amp) keeps working.
            singleton_axes = tuple(
                axis
                for axis, size in enumerate(response_shape, start=2)
                if size == 1
            )
            if singleton_axes:
                sweeped_vals = jnp.squeeze(sweeped_vals, axis=singleton_axes)
                sweeped_phase_vals = jnp.squeeze(sweeped_phase_vals, axis=singleton_axes)

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
