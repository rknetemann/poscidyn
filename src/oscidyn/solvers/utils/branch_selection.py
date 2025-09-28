import jax
import jax.numpy as jnp

def select_branches(
    t_max_disp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, n_modes)
    y_max_disp: jax.Array, # (N_COARSE_DRIVING_FREQUENCIES, N_COARSE_DRIVING_AMPLITUDES, N_COARSE_INITIAL_DISPLACEMENTS, N_COARSE_INITIAL_VELOCITIES, 2 * n_modes)
    sweep_direction: int,
) -> tuple[jax.Array, jax.Array]:

    n_coarse_freq, n_coarse_amp, n_init_disp, n_init_vel, n_modes = t_max_disp.shape
    n_state      = 2 * n_modes
    n_branches   = n_init_disp * n_init_vel

    t_branches = t_max_disp.reshape(n_coarse_freq, n_coarse_amp, n_branches, n_modes) # (n_coarse_freq, n_coarse_amp, n_branches, n_modes)
    y_branches = y_max_disp.reshape(n_coarse_freq, n_coarse_amp, n_branches, n_state) # (n_coarse_freq, n_coarse_amp, n_branches, n_modes * 2)

    is_fwd = sweep_direction == 1
    idx_fwd = jnp.arange(n_coarse_freq, dtype=int) # (n_coarse_freq,)
    idx_bwd = jnp.arange(n_coarse_freq - 1, -1, -1, dtype=int) # (n_coarse_freq,)

    def _pick_for_amplitude(t_seq, y_seq): # (n_coarse_freq, n_branches, ..)
        sweep_order = jax.lax.cond(is_fwd, lambda _: idx_fwd, lambda _: idx_bwd, operand=None)
        t_ord, y_ord = t_seq[sweep_order], y_seq[sweep_order]

        # start branch = state with min ‖y‖² at first frequency step
        idx0   = jnp.argmin(jnp.sum(y_ord[0] ** 2, axis=-1))
        y0     = y_ord[0, idx0]
        t0     = t_ord[0, idx0]

        def _step(prev_y, xy):
            t_cur, y_cur = xy
            dist  = jnp.sum((y_cur - prev_y) ** 2, axis=-1)
            idx   = jnp.argmin(dist)
            y_sel = y_cur[idx]
            t_sel = t_cur[idx]
            return y_sel, (t_sel, y_sel)

        _, (t_rest, y_rest) = jax.lax.scan(_step, y0, (t_ord[1:], y_ord[1:]))
        t_sel_ord = jnp.concatenate((t0[None, :], t_rest), axis=0)
        y_sel_ord = jnp.concatenate((y0[None, :], y_rest), axis=0)

        t_sel = jax.lax.cond(is_fwd, lambda _: t_sel_ord, lambda _: t_sel_ord[::-1], operand=None)
        y_sel = jax.lax.cond(is_fwd, lambda _: y_sel_ord, lambda _: y_sel_ord[::-1], operand=None)
        return t_sel, y_sel

    t_max_disp, y_max_disp = jax.vmap(_pick_for_amplitude, in_axes=(1, 1), out_axes=(1, 1))(t_branches, y_branches)

    return t_max_disp, y_max_disp