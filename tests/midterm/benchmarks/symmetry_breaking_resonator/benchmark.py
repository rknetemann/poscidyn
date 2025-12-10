import timeit
import time

import jax
import jax.numpy as jnp
from equinox import filter_jit
import oscidyn

BATCH_SIZE = 16
TOTAL_SINGLE_RUNS = 5
TOTAL_BATCH_RUNS = 5

@filter_jit
def simulate(params):
    """
    Simulate a Duffing system for a single set of parameters.

    `params` is a scalar. We use it to slightly perturb Q so that different
    params -> genuinely different computation, which prevents XLA from
    trivially constant-folding everything when batched.
    """
    # params has shape (), one scalar
    delta = 1e-3 * params  # small variation per simulation

    # Base parameters (use jnp so everything lives on the device)
    Q, omega_0, alpha, gamma = jnp.array([10.0, 20.0]) * (1.0 + delta), jnp.array([1.00, 2.0]), jnp.zeros((2,2,2)), jnp.zeros((2,2,2,2))
    gamma = gamma.at[0,0,0,0].set(2.67e-02)
    gamma = gamma.at[1,1,1,1].set(0.0)
    alpha = alpha.at[0,0,1].set(7.48e-01)
    alpha = alpha.at[1,0,0].set(3.74e-01)
    modal_forces = jnp.array([1.0, 0.0])

    MODEL = oscidyn.BaseDuffingOscillator(
        Q=Q,
        alpha=alpha,
        gamma=gamma,
        omega_0=omega_0,
    )

    DRIVING_FREQUENCY = jnp.linspace(0.1, 2.0, 150)
    MAX_FORCE = 0.30
    DRIVING_AMPLITUDE = jnp.linspace(0.1 * MAX_FORCE, 1.0 * MAX_FORCE, 10)
    # DRIVING_AMPLITUDE = jnp.array([MAX_FORCE])

    EXCITOR = oscidyn.OneToneExcitation(
        drive_frequencies=DRIVING_FREQUENCY,
        drive_amplitudes=DRIVING_AMPLITUDE,
        modal_forces=modal_forces,
    )

    MULTISTART = oscidyn.LinearResponseMultistart(
        init_cond_shape=(3, 3),
        linear_response_factor=1.0,
    )

    SOLVER = oscidyn.TimeIntegrationSolver(
        max_steps=4096,
        n_time_steps=50,
        verbose=False,
        throw=False,
        rtol=1e-4,
        atol=1e-7,
    )

    SWEEPER = oscidyn.NearestNeighbourSweep(
        sweep_direction=[oscidyn.Forward(), oscidyn.Backward()]
    )

    PRECISION = oscidyn.Precision.SINGLE

    frequency_sweep = oscidyn.frequency_sweep(
        model=MODEL,
        sweeper=SWEEPER,
        excitor=EXCITOR,
        solver=SOLVER,
        precision=PRECISION,
        multistarter=MULTISTART,
    )

    return frequency_sweep


def benchmark_single_frequency_sweep():
    """
    One JIT-compiled simulation (no batching).
    Called many times by timeit.
    """
    # Single scalar param (you can randomize it if you like)
    param = jnp.array(0.5)

    out = simulate(param)
    jax.block_until_ready(out)


def benchmark_batch_frequency_sweep():
    """
    One batched simulation of size BATCH_SIZE using vmap over `simulate`.
    Called many times by timeit.
    """
    # Different params per lane so the work can't trivially be broadcast
    batch_params = jnp.linspace(0.0, 1.0, BATCH_SIZE)  # shape (BATCH_SIZE,)

    batched_simulate = jax.vmap(simulate)
    out = batched_simulate(batch_params)
    jax.block_until_ready(out)


if __name__ == "__main__":
    # ---- WARMUP (compilation happens here, excluded from timings) ----

    # Warmup single-run path
    start_time = time.time()
    benchmark_single_frequency_sweep()
    end_time = time.time()
    time_single_frequency_sweep_with_compilation = end_time - start_time
    print(f"Warmup single-run path took {time_single_frequency_sweep_with_compilation:.4f} seconds.")

    # Warmup batched path
    start_time = time.time()
    benchmark_batch_frequency_sweep()
    end_time = time.time()
    time_batch_frequency_sweep_with_compilation = end_time - start_time
    print(f"Warmup batched path took {time_batch_frequency_sweep_with_compilation:.4f} seconds.")

    # ---- BENCHMARK SINGLE RUNS ----
    single_time = timeit.timeit(
        number=TOTAL_SINGLE_RUNS,
        stmt=benchmark_single_frequency_sweep,
    )

    print(f"[SINGLE] Total time for {TOTAL_SINGLE_RUNS} runs: "
          f"{single_time:.4f} s")
    print(f"[SINGLE] Per run: {single_time / TOTAL_SINGLE_RUNS:.6f} s")

    # ---- BENCHMARK BATCHED RUNS ----
    batch_time = timeit.timeit(
        number=TOTAL_BATCH_RUNS,
        stmt=benchmark_batch_frequency_sweep,
    )

    print(f"\n[BATCHED] Total time for {TOTAL_BATCH_RUNS} batches "
          f"of size {BATCH_SIZE}: {batch_time:.4f} s")
    print(f"[BATCHED] Per batch: {batch_time / TOTAL_BATCH_RUNS:.6f} s")
    print(f"[BATCHED] Approx per simulation in batch: "
          f"{(batch_time / TOTAL_BATCH_RUNS) / BATCH_SIZE:.6f} s")

    # ---- OPTIONAL: ratio of per-simulation costs ----
    per_single = single_time / TOTAL_SINGLE_RUNS
    per_batched = (batch_time / TOTAL_BATCH_RUNS) / BATCH_SIZE
    print(f"\nSpeedup factor (single per sim / batched per sim): "
          f"{per_single / per_batched:.3f}x")
