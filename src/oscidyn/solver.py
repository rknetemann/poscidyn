# solver.py
import jax
import jax.numpy as jnp
import diffrax
import optimistix as optx

from .models import AbstractModel
from . import constants as const            # needs N_PERIODS_TO_RETAIN

jax.config.update("jax_enable_x64", False)
jax.config.update('jax_platform_name', 'gpu')

# import os
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
# )

class AbstractSolver:
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096):
        self.n_time_steps = n_time_steps
        self.max_steps = max_steps

class StandardSolver(AbstractSolver):
    def __init__(self, t_end: float, n_time_steps: int = 2000, max_steps: int = 4096):
        super().__init__(n_time_steps)
        self.t_end = t_end
        self.max_steps = max_steps

    def solve_rhs(self, model: AbstractModel, driving_frequency: jax.Array, driving_amplitude: jax.Array, initial_condition: jax.Array) -> jax.Array:
        sol = diffrax.diffeqsolve(
            terms=diffrax.ODETerm(model.rhs_jit),
            solver=diffrax.Tsit5(),
            t0=0.0,
            t1=self.t_end,
            dt0=None,
            max_steps=self.max_steps,
            y0=initial_condition,
            throw=True,
            progress_meter=diffrax.TqdmProgressMeter(),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, self.t_end, self.n_time_steps)),
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
            args=(driving_frequency, driving_amplitude),
        )

        time = sol.ts
        displacements = sol.ys[:, : model.n_modes]
        velocities = sol.ys[:, model.n_modes :]
        
        return time, displacements, velocities

class SteadyStateSolver(AbstractSolver):
    def __init__(self, n_time_steps: int = 2000, max_steps: int = 4096, tol: float = 1e-6):
        super().__init__(n_time_steps)
        self.max_steps = max_steps
        self.tol = tol
    
    def solve_rhs(
        self,
        model: AbstractModel,
        driving_frequency: jax.Array,     # scalar
        driving_amplitude:  jax.Array,     # scalar
        initial_condition:  jax.Array,     # shape (2*n_modes,)
    ):
        """
        Integrate one drive period at a time until the RMS displacement
        between successive periods changes by less than `self.tol`
        (both absolute and relative).  Works with `vmap`/`jit`.
        """
        # ---------------- static info ---------------------------------
        T            = 2.0 * jnp.pi / driving_frequency          # drive period
        n_pts_period = self.n_time_steps                         # samples / period
        n_modes      = model.n_modes
        state_dim    = 2 * n_modes

        # safety cap (set in __init__)
        max_periods  = getattr(self, "max_periods", 512)

        ts_rel = jnp.linspace(0.0, T, n_pts_period)              # (n_pts_period,)

        # ring buffer stores the last const.N_PERIODS_TO_RETAIN periods
        buf_t = jnp.zeros((const.N_PERIODS_TO_RETAIN, n_pts_period),
                          dtype=initial_condition.dtype)
        buf_y = jnp.zeros((const.N_PERIODS_TO_RETAIN,
                           n_pts_period, state_dim),
                          dtype=initial_condition.dtype)

        # carry = (t0, y0, prev_rms, delta_rms, k, buf_t, buf_y)
        carry0 = (
            jnp.array(0.0, dtype=initial_condition.dtype),       # t0
            initial_condition,                                   # y0
            jnp.array(jnp.inf, dtype=initial_condition.dtype),   # prev_rms
            jnp.array(jnp.inf, dtype=initial_condition.dtype),   # delta_rms
            jnp.array(0, dtype=jnp.int32),                       # k
            buf_t,
            buf_y,
        )

        # ---------------- loop body ----------------------------------
        def body(carry):
            t0, y0, prev_rms, _delta, k, buf_t, buf_y = carry
            ts_abs = t0 + ts_rel

            # sensible first step → roughly one output spacing
            dt0   = T / n_pts_period

            sol = diffrax.diffeqsolve(
                terms=diffrax.ODETerm(model.rhs_jit),
                solver=diffrax.Tsit5(),
                t0=t0,
                t1=t0 + T,
                dt0=dt0,
                y0=y0,
                max_steps=self.max_steps,
                saveat=diffrax.SaveAt(ts=ts_abs),
                stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
                args=(driving_frequency, driving_amplitude),
                throw=False,                                     # avoid exception on max_steps
            )

            # compute RMS displacement over this period
            disp = sol.ys[:, :n_modes]                           # (n_pts, n_modes)
            rms  = jnp.sqrt(jnp.mean(disp**2))                   # scalar
            delta_rms = jnp.abs(rms - prev_rms)

            # update ring buffer
            slot  = k % const.N_PERIODS_TO_RETAIN
            buf_t = buf_t.at[slot].set(sol.ts)
            buf_y = buf_y.at[slot].set(sol.ys)

            return (
                t0 + T,              # next period start
                sol.ys[-1],          # next initial state
                rms,                 # new prev_rms
                delta_rms,           # new delta for cond check
                k + 1,
                buf_t,
                buf_y,
            )

        # ---------------- loop condition -----------------------------
        def cond(carry):
            _t0, _y0, prev_rms, delta_rms, k, *_ = carry
            # convergence test: both absolute and relative
            rel_ok  = delta_rms / jnp.maximum(prev_rms, 1e-12) < 5 * self.tol
            abs_ok  = delta_rms < self.tol
            done    = jnp.logical_and(rel_ok, abs_ok)
            not_max = k < max_periods
            return jnp.logical_and(~done, not_max)

        # ---------------- run loop -----------------------------------
        t0_f, y0_f, rms_f, delta_f, k_f, buf_t_f, buf_y_f = \
            jax.lax.while_loop(cond, body, carry0)

        # sanity: raise if we never converged
        if max_periods and isinstance(k_f, jax.Array):
            # (inside JIT this becomes a host callback)
            def _host_assert(k_val):
                if k_val == max_periods:
                    raise RuntimeError(
                        f"Steady‑state not reached after {max_periods} periods "
                        f"(tol={self.tol}, last Δrms={delta_f}).")
                return ()
            jax.debug.callback(_host_assert, k_f, ordered=True)

        # reorder ring buffer chronologically
        order = jnp.mod(jnp.arange(k_f - const.N_PERIODS_TO_RETAIN, k_f),
                        const.N_PERIODS_TO_RETAIN)
        time  = buf_t_f[order].reshape(-1)               # (Nret,)
        y_all = buf_y_f[order].reshape(-1, state_dim)    # (Nret, 2*n_modes)

        displacements = y_all[:, :n_modes]
        velocities    = y_all[:, n_modes:]

        return time, displacements, velocities