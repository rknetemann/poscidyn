import argparse
import time

import jax
import jax.numpy as jnp
import optimistix as optx
import equinox as eqx


# Often import when doing scientific work
jax.config.update("jax_enable_x64", True)

STEADY_STATE = jnp.array(
    [0.8, -0.35, 0.5, -0.9, 0.65, -0.25, 0.15],
    dtype=jnp.float64,
)

@eqx.filter_jit
def _nonlinear_coupling(vec: jnp.ndarray) -> jnp.ndarray:
    """Seven-variable coupled system with strong nonlinearities."""
    v1, v2, v3, v4, v5, v6, v7 = vec
    eq1 = jnp.tanh(v1 * v2) + 0.2 * v5**2 + jnp.exp(-v7) + v3 * v4
    eq2 = jnp.sin(v2 + v3) + 0.5 * v6 - v1 * v7
    eq3 = jnp.log1p(v3**2 + v4**2) + v2 * v5
    eq4 = v4 + jnp.sinh(v1 - v6) - 0.3 * v7**2
    eq5 = jnp.cos(v5 - v2) + v6 * v7 + 0.1 * v1**2
    eq6 = jnp.arctan(v6 * v3) + v4 * v5 - jnp.tanh(v2)
    eq7 = v1 + v2 + v3 - v4 + jnp.sin(v5 + v6 + v7)
    return jnp.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7], dtype=jnp.float64)


REFERENCE_STATE = _nonlinear_coupling(STEADY_STATE)

@eqx.filter_jit
def test():
    def fn(y, args):
        return _nonlinear_coupling(y) - REFERENCE_STATE

    solver = optx.Newton(rtol=1e-8, atol=1e-8)
    perturbation = jnp.array(
        [0.15, -0.1, 0.05, 0.25, -0.2, 0.3, -0.1],
        dtype=jnp.float64,
    )
    y0 = STEADY_STATE + perturbation
    sol = optx.root_find(fn, solver, y0)

    return jax.device_get(sol.value)


def benchmark(iterations: int = 100, warmup: int = 5) -> None:
    """Run `test` repeatedly and report the mean runtime."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    # Warm up to exclude compilation time from measurements.
    for _ in range(warmup):
        test()

    total = 0.0
    last_value = None
    for _ in range(iterations):
        start = time.perf_counter()
        last_value = test()
        total += time.perf_counter() - start

    avg = total / iterations
    print(
        f"Ran {iterations} iterations (after {warmup} warmups). "
        f"Average runtime: {avg * 1e3:.3f} ms"
    )
    print(f"Last value: {last_value}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the optimistix root-finding test case."
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=10000,
        help="number of timed runs to average (default: 100)",
    )
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=5,
        help="number of warmup runs to exclude from timing (default: 5)",
    )
    args = parser.parse_args()
    benchmark(iterations=args.iterations, warmup=args.warmup)


if __name__ == "__main__":
    main()
