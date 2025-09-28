import jax
import jax.numpy as jnp

def get_unique_solutions(
    y0: jax.Array,
    max_branches: int,
    n: int,
    n_freq: int,
    n_amp: int,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    y0: (n_sim, n) with n_sim = n_freq * n_amp * n_ic
    Returns:
      y0_unique:       (max_branches, n, n_freq, n_amp)  # NaN-padded where unused
      branches_count:  (n_freq, n_amp)                   # number of unique branches kept in each cell
      counts_per_slot: (max_branches, n_freq, n_amp)     # how many rows mapped to each kept branch
    """
    assert y0.ndim == 2 and y0.shape[1] == n, "y0 shape must be (n_sim, n)"
    total_cells = n_freq * n_amp
    assert y0.shape[0] % total_cells == 0, "y0 rows must be divisible by n_freq * n_amp"
    n_ic = y0.shape[0] // total_cells

    # Group rows by (freq, amp) -> shape (n_freq, n_amp, n_ic, n)
    y_grp = y0.reshape(n_freq, n_amp, n_ic, n)

    def per_group(y_group: jax.Array):
        """
        y_group: (n_ic, n)
        Select up to max_branches unique rows under allclose(rtol, atol),
        keeping first occurrences. Returns NaN-padded reps and bookkeeping.
        """
        reps   = jnp.zeros((max_branches, n), dtype=y_group.dtype)          # slots
        valid  = jnp.zeros((max_branches,), dtype=bool)
        counts = jnp.zeros((max_branches,), dtype=jnp.int32)
        M      = jnp.int32(0)  # number of reps filled

        def body(i, carry):
            reps, valid, counts, M = carry
            curr = y_group[i]  # (n,)

            # Compare to existing reps (only where valid=True)
            ref  = jnp.maximum(jnp.abs(reps), jnp.abs(curr))                 # (max_branches, n)
            tol  = atol + rtol * ref
            close = jnp.all(jnp.abs(reps - curr) <= tol, axis=1) & valid     # (max_branches,)

            candidates  = jnp.where(close, jnp.arange(max_branches, dtype=jnp.int32), max_branches)
            first_match = jnp.min(candidates)
            matched     = first_match < max_branches
            has_space   = M < max_branches
            create_new  = (~matched) & has_space
            slot        = jnp.where(matched, first_match, M)

            reps  = jax.lax.cond(create_new, lambda r: r.at[slot].set(curr), lambda r: r, reps)
            valid = jax.lax.cond(create_new, lambda v: v.at[slot].set(True),  lambda v: v, valid)
            counts = counts.at[slot].add(1)
            M = jax.lax.select(create_new, M + 1, M)
            return reps, valid, counts, M

        reps, valid, counts, M = jax.lax.fori_loop(0, y_group.shape[0], body,
                                                   (reps, valid, counts, jnp.int32(0)))

        # Pad unused slots with NaN for clarity
        reps = jnp.where(valid[:, None], reps, jnp.array(jnp.nan, reps.dtype))

        return reps, valid, counts, M

    # Vectorize over (freq, amp)
    process_all = jax.jit(
        jax.vmap(
            jax.vmap(per_group, in_axes=0, out_axes=(0, 0, 0, 0)),  # over n_amp
            in_axes=0, out_axes=(0, 0, 0, 0)                        # over n_freq
        )
    )

    reps, valid, counts, M = process_all(y_grp)
    # reps:   (n_freq, n_amp, max_branches, n)
    # valid:  (n_freq, n_amp, max_branches)  (not returned, but could be)
    # counts: (n_freq, n_amp, max_branches)
    # M:      (n_freq, n_amp)

    # Reorder to requested layout
    y0_unique      = jnp.transpose(reps, (2, 3, 0, 1))            # (max_branches, n, n_freq, n_amp)
    counts_per_slot = jnp.transpose(counts, (2, 0, 1))            # (max_branches, n_freq, n_amp)
    branches_count  = M                                            # (n_freq, n_amp)

    return y0_unique, branches_count, counts_per_slot
