import timeit

import jax
import jax.numpy as jnp
from equinox import filter_jit
import poscidyn  # your library


BATCH_SIZE = 4
TOTAL_BATCHES = 10


@filter_jit
def simulate(params):
    """
    Simulate a Duffing system for a single set of parameters.

    `params` is a scalar per vmapped lane. We use it to perturb Q slightly so
    each batch element is genuinely different and cannot be constant-folded.
    """
    # params has shape (), one scalar per vmap lane.
    delta = 1e-3 * params  # small variation per batch element

    # Base parameters (use jnp so everything is on-device)
    Q = jnp.array([10.0, 20.0]) * (1.0 + delta)
    omega_0 = jnp.array([1.00, 2.0])
    alpha = jnp.zeros((2, 2, 2))
    gamma = jnp.zeros((2, 2, 2, 2))

    gamma = gamma.at[0, 0, 0, 0].set(2.67e-02)
    gamma = gamma.at[1, 1, 1, 1].set(5.40e-01)
    alpha = alpha.at[0, 0, 1].set(7.48e-01)
    alpha = alpha.at[1, 0, 0].set(3.74e-01)

    MODEL = poscidyn.BaseDuffingOscillator(
        Q=Q,
        alpha=alpha,
        gamma=gamma,
        omega_0=omega_0,
    )

    DRIVING_FREQUENCY = jnp.linspace(0.5, 1.5, 150)
    MAX_FORCE = 0.3
    DRIVING_AMPLITUDE = jnp.linspace(0.1 * MAX_FORCE, 1.0 * MAX_FORCE, 10)

    EXCITOR = poscidyn.OneToneExcitation(
        drive_frequencies=DRIVING_FREQUENCY,
        drive_amplitudes=DRIVING_AMPLITUDE,
        modal_forces=jnp.array([1.0, 1.0]),
    )

    MULTISTART = poscidyn.LinearResponseMultistart(
        init_cond_shape=(3, 3),
        linear_response_factor=1.0,
    )

    SOLVER = poscidyn.TimeIntegrationSolver(
        max_steps=4096 * 1,
        n_time_steps=50,
        verbose=False,  # usually disable for benchmarking
        throw=False,
        rtol=1e-3,
        atol=1e-7,
    )

    SWEEPER = poscidyn.NearestNeighbourSweep(
        sweep_direction=[poscidyn.Forward(), poscidyn.Backward()]
    )

    PRECISION = poscidyn.Precision.SINGLE

    frequency_sweep = poscidyn.frequency_sweep(
        model=MODEL,
        sweeper=SWEEPER,
        excitor=EXCITOR,
        solver=SOLVER,
        precision=PRECISION,
        multistarter=MULTISTART,
    )

    # Return JAX arrays so we can block on the actual computation.
    return frequency_sweep


def benchmark_batch_frequency_sweep():
    """
    Run a batched simulation of size BATCH_SIZE and block until all work is done.
    This is the function we time with timeit.
    """
    # Make per-lane params genuinely different
    batch_params = jnp.linspace(0.0, 1.0, BATCH_SIZE)  # shape (BATCH_SIZE,)

    # vmap over the leading dimension of batch_params
    batch_sweeps = jax.vmap(simulate)(batch_params)

    # Force all queued device work to finish before returning
    jax.block_until_ready(batch_sweeps)


if __name__ == "__main__":
    # Warmup: compile once and run, so compilation cost is not measured
    benchmark_batch_frequency_sweep()

    # Benchmark TOTAL_BATCHES batched runs
    execution_time = timeit.timeit(
        number=TOTAL_BATCHES,
        stmt=benchmark_batch_frequency_sweep,
    )

    print(f"Benchmark completed in {execution_time:.4f} seconds "
          f"for {TOTAL_BATCHES} batches of size {BATCH_SIZE}.")
    print(f"Per batch: {execution_time / TOTAL_BATCHES:.6f} seconds")
    print(f"Per simulation (approx): "
          f"{(execution_time / TOTAL_BATCHES) / BATCH_SIZE:.6f} seconds")
