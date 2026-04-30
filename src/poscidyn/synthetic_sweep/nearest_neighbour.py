from .abstract_synthetic_sweep import AbstractSyntheticSweep, AbstractSyntheticSweepDirection, Forward, Backward
import jax.numpy as jnp
from equinox import filter_jit
from jax import lax

class NearestNeighbour(AbstractSyntheticSweep):
    def __init__(
        self,
        sweep_direction: AbstractSyntheticSweepDirection | list[AbstractSyntheticSweepDirection] = [Forward(), Backward()],
        phase_weight: float = 0.25,
        seed_switch_penalty: float = 0.05,
    ):
        super().__init__()
        self.sweep_direction = sweep_direction
        self.phase_weight = phase_weight
        self.seed_switch_penalty = seed_switch_penalty

    def to_dtype(self, dtype):
        return NearestNeighbour(
            sweep_direction=self.sweep_direction,
            phase_weight=self.phase_weight,
            seed_switch_penalty=self.seed_switch_penalty,
        )

    @filter_jit
    def sweep(self, periodic_solutions):
        modal_amplitudes = periodic_solutions["modal_amplitude"]
        modal_phases = periodic_solutions["modal_phase"]
        modal_response_frequencies = periodic_solutions.get("modal_response_frequency")
        if modal_response_frequencies is None:
            modal_response_frequencies = jnp.full_like(modal_phases, jnp.nan)

        total_amplitudes = periodic_solutions["total_amplitude"]
        total_phases = periodic_solutions["total_phase"]
        total_response_frequencies = periodic_solutions.get("total_response_frequency")
        if total_response_frequencies is None:
            total_response_frequencies = jnp.full_like(total_phases, jnp.nan)

        n_freq, n_amp, n_init_disp, n_init_vel = modal_amplitudes.shape[:4]
        modal_shape = modal_amplitudes.shape[4:]
        total_shape = total_amplitudes.shape[4:]
        n_seeds = n_init_disp * n_init_vel

        modal_amplitudes_seed = modal_amplitudes.reshape((n_freq, n_amp, n_seeds) + modal_shape)
        modal_phases_seed = modal_phases.reshape((n_freq, n_amp, n_seeds) + modal_shape)
        modal_response_frequencies_seed = modal_response_frequencies.reshape(
            (n_freq, n_amp, n_seeds) + modal_shape
        )
        total_amplitudes_seed = total_amplitudes.reshape((n_freq, n_amp, n_seeds) + total_shape)
        total_phases_seed = total_phases.reshape((n_freq, n_amp, n_seeds) + total_shape)
        total_response_frequencies_seed = total_response_frequencies.reshape(
            (n_freq, n_amp, n_seeds) + total_shape
        )

        results = {
            "forward": None,
            "backward": None,
            "forward_phase": None,
            "backward_phase": None,
            "forward_demod_freq": None,
            "backward_demod_freq": None,
            "forward_total": None,
            "backward_total": None,
            "forward_total_phase": None,
            "backward_total_phase": None,
            "forward_total_demod_freq": None,
            "backward_total_demod_freq": None,
            "forward_idx": None,
            "backward_idx": None,
        }

        def _unwrap_if_phase_in_rad(arr):
            finite_phase_mask = jnp.isfinite(arr)
            max_abs_phase = jnp.max(jnp.abs(jnp.where(finite_phase_mask, arr, 0.0)))
            has_finite_phase = jnp.any(finite_phase_mask)
            should_unwrap = jnp.logical_and(has_finite_phase, max_abs_phase <= (jnp.pi + 1e-6))
            return jnp.where(
                should_unwrap,
                jnp.unwrap(arr, axis=0),
                arr,
            )

        for direction_obj in (self.sweep_direction if isinstance(self.sweep_direction, list) else [self.sweep_direction]):
            if isinstance(direction_obj, Forward):
                order = jnp.arange(n_freq)
                start_idx = 0
                direction = "forward"
            elif isinstance(direction_obj, Backward):
                order = jnp.arange(n_freq - 1, -1, -1)
                start_idx = n_freq - 1
                direction = "backward"
            else:
                raise ValueError("Unsupported sweep direction")

            def sweep_one_amplitude(amp_idx):
                start_modal = modal_amplitudes_seed[start_idx, amp_idx]
                start_modal_phase = modal_phases_seed[start_idx, amp_idx]
                start_modal_freq = modal_response_frequencies_seed[start_idx, amp_idx]
                start_total = total_amplitudes_seed[start_idx, amp_idx]
                start_total_phase = total_phases_seed[start_idx, amp_idx]
                start_total_freq = total_response_frequencies_seed[start_idx, amp_idx]

                # Medoid selection using amplitude + phase cost (consistent with per-step cost).
                start_modal_flat = start_modal.reshape((n_seeds, -1))
                start_modal_phase_flat = start_modal_phase.reshape((n_seeds, -1))
                finite_mask = jnp.all(jnp.isfinite(start_modal_flat), axis=1)

                # Pairwise RMS amplitude distance, normalised by the candidate's own RMS scale.
                pairwise_amp_diffs = start_modal_flat[:, None, :] - start_modal_flat[None, :, :]
                pairwise_amp_dist = jnp.sqrt(jnp.mean(pairwise_amp_diffs ** 2, axis=-1))  # (n_seeds, n_seeds)
                pairwise_amp_scale = jnp.sqrt(jnp.mean(start_modal_flat ** 2, axis=-1)) + 1e-12  # (n_seeds,)
                pairwise_amp_cost = pairwise_amp_dist / pairwise_amp_scale[:, None]  # normalise by the "reference" row

                # Pairwise circular phase distance.
                pairwise_phase_delta = jnp.angle(
                    jnp.exp(1j * (start_modal_phase_flat[:, None, :] - start_modal_phase_flat[None, :, :]))
                )
                phase_finite = jnp.isfinite(start_modal_phase_flat)
                pairwise_phase_finite = jnp.logical_and(phase_finite[:, None, :], phase_finite[None, :, :])
                phase_count = jnp.sum(pairwise_phase_finite, axis=-1)  # (n_seeds, n_seeds)
                safe_phase_count = jnp.maximum(phase_count, 1)
                pairwise_phase_rms = jnp.sqrt(
                    jnp.sum(jnp.where(pairwise_phase_finite, pairwise_phase_delta ** 2, 0.0), axis=-1)
                    / safe_phase_count
                )
                pairwise_phase_cost = pairwise_phase_rms / jnp.pi

                pairwise_cost = pairwise_amp_cost + self.phase_weight * pairwise_phase_cost  # (n_seeds, n_seeds)

                valid_pair_mask = jnp.logical_and(finite_mask[:, None], finite_mask[None, :])
                pairwise_cost_masked = jnp.where(valid_pair_mask, pairwise_cost, 0.0)

                valid_counts = jnp.sum(valid_pair_mask, axis=1)
                safe_valid_counts = jnp.maximum(valid_counts, 1)
                mean_cost_to_others = jnp.sum(pairwise_cost_masked, axis=1) / safe_valid_counts
                safe_mean_cost_to_others = jnp.where(finite_mask, mean_cost_to_others, jnp.inf)

                start_choice_idx_raw = jnp.argmin(safe_mean_cost_to_others)
                has_valid_start = jnp.any(finite_mask)

                start_modal_choice = jnp.where(
                    has_valid_start,
                    start_modal[start_choice_idx_raw],
                    jnp.zeros(modal_shape, dtype=modal_amplitudes.dtype),
                )
                start_choice_idx = jnp.where(
                    has_valid_start,
                    start_choice_idx_raw,
                    -1,
                )
                safe_start_idx = jnp.maximum(start_choice_idx, 0)
                start_modal_phase_choice = jnp.where(
                    has_valid_start,
                    start_modal_phase[safe_start_idx],
                    jnp.full(modal_shape, jnp.nan, dtype=modal_phases.dtype),
                )
                start_modal_freq_choice = jnp.where(
                    has_valid_start,
                    start_modal_freq[safe_start_idx],
                    jnp.full(modal_shape, jnp.nan, dtype=modal_response_frequencies.dtype),
                )
                start_total_choice = jnp.where(
                    has_valid_start,
                    start_total[safe_start_idx],
                    jnp.full(total_shape, jnp.nan, dtype=total_amplitudes.dtype),
                )
                start_total_phase_choice = jnp.where(
                    has_valid_start,
                    start_total_phase[safe_start_idx],
                    jnp.full(total_shape, jnp.nan, dtype=total_phases.dtype),
                )
                start_total_freq_choice = jnp.where(
                    has_valid_start,
                    start_total_freq[safe_start_idx],
                    jnp.full(total_shape, jnp.nan, dtype=total_response_frequencies.dtype),
                )

                def body(carry, freq_idx):
                    (
                        prev_modal,
                        prev_modal_phase,
                        prev_modal_freq,
                        prev_total,
                        prev_total_phase,
                        prev_total_freq,
                        prev_idx,
                        out_modal,
                        out_modal_phase,
                        out_modal_freq,
                        out_total,
                        out_total_phase,
                        out_total_freq,
                        out_idx,
                    ) = carry

                    modal_cands = modal_amplitudes_seed[freq_idx, amp_idx]
                    modal_phase_cands = modal_phases_seed[freq_idx, amp_idx]
                    modal_freq_cands = modal_response_frequencies_seed[freq_idx, amp_idx]
                    total_cands = total_amplitudes_seed[freq_idx, amp_idx]
                    total_phase_cands = total_phases_seed[freq_idx, amp_idx]
                    total_freq_cands = total_response_frequencies_seed[freq_idx, amp_idx]

                    modal_cands_flat = modal_cands.reshape((n_seeds, -1))
                    prev_modal_flat = prev_modal.reshape((1, -1))
                    modal_phase_cands_flat = modal_phase_cands.reshape((n_seeds, -1))
                    prev_modal_phase_flat = prev_modal_phase.reshape((1, -1))

                    # RMS amplitude distance over all response components.
                    amp_diffs_flat = modal_cands_flat - prev_modal_flat
                    amp_diffs = jnp.sqrt(jnp.mean(amp_diffs_flat**2, axis=1))  # (n_seeds,)
                    prev_amp_scale = jnp.sqrt(jnp.mean(prev_modal_flat**2)) + 1e-12
                    amp_cost = amp_diffs / prev_amp_scale

                    # Circular phase distance; ignored where phase is not finite.
                    phase_delta = jnp.angle(
                        jnp.exp(1j * (modal_phase_cands_flat - prev_modal_phase_flat))
                    )
                    phase_finite_mask = jnp.logical_and(
                        jnp.isfinite(modal_phase_cands_flat),
                        jnp.isfinite(prev_modal_phase_flat),
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
                    finite_mask = jnp.all(jnp.isfinite(modal_cands_flat), axis=1)
                    diffs = jnp.where(finite_mask, diffs, jnp.inf)

                    k = jnp.argmin(diffs)
                    no_valid = jnp.logical_not(jnp.any(finite_mask))
                    chosen_modal = jnp.where(no_valid, prev_modal, modal_cands[k])
                    chosen_modal_phase = jnp.where(no_valid, prev_modal_phase, modal_phase_cands[k])
                    chosen_modal_freq = jnp.where(no_valid, prev_modal_freq, modal_freq_cands[k])
                    chosen_total = jnp.where(no_valid, prev_total, total_cands[k])
                    chosen_total_phase = jnp.where(no_valid, prev_total_phase, total_phase_cands[k])
                    chosen_total_freq = jnp.where(no_valid, prev_total_freq, total_freq_cands[k])
                    chosen_idx = jnp.where(no_valid, prev_idx, k)

                    out_modal = out_modal.at[freq_idx].set(chosen_modal)
                    out_modal_phase = out_modal_phase.at[freq_idx].set(chosen_modal_phase)
                    out_modal_freq = out_modal_freq.at[freq_idx].set(chosen_modal_freq)
                    out_total = out_total.at[freq_idx].set(chosen_total)
                    out_total_phase = out_total_phase.at[freq_idx].set(chosen_total_phase)
                    out_total_freq = out_total_freq.at[freq_idx].set(chosen_total_freq)
                    out_idx = out_idx.at[freq_idx].set(chosen_idx)
                    return (
                        chosen_modal,
                        chosen_modal_phase,
                        chosen_modal_freq,
                        chosen_total,
                        chosen_total_phase,
                        chosen_total_freq,
                        chosen_idx,
                        out_modal,
                        out_modal_phase,
                        out_modal_freq,
                        out_total,
                        out_total_phase,
                        out_total_freq,
                        out_idx,
                    ), None

                init_modal = jnp.full((n_freq,) + modal_shape, jnp.nan, dtype=modal_amplitudes.dtype)
                init_modal_phase = jnp.full((n_freq,) + modal_shape, jnp.nan, dtype=modal_phases.dtype)
                init_modal_freq = jnp.full(
                    (n_freq,) + modal_shape, jnp.nan, dtype=modal_response_frequencies.dtype
                )
                init_total = jnp.full((n_freq,) + total_shape, jnp.nan, dtype=total_amplitudes.dtype)
                init_total_phase = jnp.full((n_freq,) + total_shape, jnp.nan, dtype=total_phases.dtype)
                init_total_freq = jnp.full(
                    (n_freq,) + total_shape, jnp.nan, dtype=total_response_frequencies.dtype
                )
                init_idx = jnp.full((n_freq,), -1, dtype=jnp.int32)

                (
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    out_modal,
                    out_modal_phase,
                    out_modal_freq,
                    out_total,
                    out_total_phase,
                    out_total_freq,
                    out_idx,
                ), _ = lax.scan(
                    body,
                    (
                        start_modal_choice,
                        start_modal_phase_choice,
                        start_modal_freq_choice,
                        start_total_choice,
                        start_total_phase_choice,
                        start_total_freq_choice,
                        start_choice_idx,
                        init_modal,
                        init_modal_phase,
                        init_modal_freq,
                        init_total,
                        init_total_phase,
                        init_total_freq,
                        init_idx,
                    ),
                    order,
                )
                return (
                    out_modal,
                    out_modal_phase,
                    out_modal_freq,
                    out_total,
                    out_total_phase,
                    out_total_freq,
                    out_idx,
                )

            (
                modal_vals,
                modal_phase_vals,
                modal_freq_vals,
                total_vals,
                total_phase_vals,
                total_freq_vals,
                idxs,
            ) = zip(*(sweep_one_amplitude(amp) for amp in range(n_amp)))

            sweeped_modal = jnp.stack(modal_vals, axis=1)
            sweeped_modal_phase = jnp.stack(modal_phase_vals, axis=1)
            sweeped_modal_freq = jnp.stack(modal_freq_vals, axis=1)
            sweeped_total = jnp.stack(total_vals, axis=1)
            sweeped_total_phase = jnp.stack(total_phase_vals, axis=1)
            sweeped_total_freq = jnp.stack(total_freq_vals, axis=1)

            modal_singleton_axes = tuple(
                axis for axis, size in enumerate(modal_shape, start=2) if size == 1
            )
            total_singleton_axes = tuple(
                axis for axis, size in enumerate(total_shape, start=2) if size == 1
            )
            if modal_singleton_axes:
                sweeped_modal = jnp.squeeze(sweeped_modal, axis=modal_singleton_axes)
                sweeped_modal_phase = jnp.squeeze(sweeped_modal_phase, axis=modal_singleton_axes)
                sweeped_modal_freq = jnp.squeeze(sweeped_modal_freq, axis=modal_singleton_axes)
            if total_singleton_axes:
                sweeped_total = jnp.squeeze(sweeped_total, axis=total_singleton_axes)
                sweeped_total_phase = jnp.squeeze(sweeped_total_phase, axis=total_singleton_axes)
                sweeped_total_freq = jnp.squeeze(sweeped_total_freq, axis=total_singleton_axes)

            sweeped_modal_phase = _unwrap_if_phase_in_rad(sweeped_modal_phase)
            sweeped_total_phase = _unwrap_if_phase_in_rad(sweeped_total_phase)
            sweeped_idxs = jnp.stack(idxs, axis=1)  # (n_freq, n_amp)

            results[direction] = sweeped_modal
            results[f"{direction}_phase"] = sweeped_modal_phase
            results[f"{direction}_demod_freq"] = sweeped_modal_freq
            results[f"{direction}_total"] = sweeped_total
            results[f"{direction}_total_phase"] = sweeped_total_phase
            results[f"{direction}_total_demod_freq"] = sweeped_total_freq
            results[f"{direction}_idx"] = sweeped_idxs

        return results
