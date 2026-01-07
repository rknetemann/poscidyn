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
        Expects periodic_solutions['max_x_total'] with shape:
        (n_freq, n_amp, n_init_disp, n_init_vel) and entries being scalar
        response amplitudes for each (init_disp, init_vel) seed.
        Returns:
          sweeped_vals: (n_freq, n_amp) chosen amplitude per freq & amp
          sweeped_idx:  (n_freq, n_amp) flat index of the chosen seed (debug)
        """
        max_x_total = periodic_solutions['max_x_total']  # scalar amplitudes
        n_freq, n_amp, _, _ = max_x_total.shape

        results = {'forward': None, 'backward': None}

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
                start_cands = max_x_total[start_idx, amp_idx].reshape(-1)

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

                def body(carry, freq_idx):
                    prev_val, out_vals, out_idx = carry
                    cands = max_x_total[freq_idx, amp_idx].reshape(-1)  # (n_seeds,)
                    # distance in *amplitude* space
                    diffs = jnp.abs(cands - prev_val)

                    # ignore NaNs/Infs by giving them infinite distance
                    finite_mask = jnp.isfinite(cands)
                    diffs = jnp.where(finite_mask, diffs, jnp.inf)

                    k = jnp.argmin(diffs)
                    no_valid = jnp.logical_not(jnp.any(finite_mask))
                    chosen = jnp.where(no_valid, prev_val, cands[k])
                    chosen_idx = jnp.where(no_valid, -1, k)

                    out_vals = out_vals.at[freq_idx].set(chosen)
                    out_idx = out_idx.at[freq_idx].set(chosen_idx)
                    return (chosen, out_vals, out_idx), None

                init_vals = jnp.full((n_freq,), jnp.nan)
                init_idx = jnp.full((n_freq,), -1, dtype=jnp.int32)

                (final_val, out_vals, out_idx), _ = lax.scan(
                    body, (start_choice, init_vals, init_idx), order
                )
                return out_vals, out_idx

            vals, idxs = zip(*(sweep_one_amplitude(amp) for amp in range(n_amp)))
            sweeped_vals = jnp.stack(vals, axis=1)  # (n_freq, n_amp)
            results[direction] = sweeped_vals

        return results
