import jax
import numpy as np
from equinox import filter_jit
import oscidyn
import timeit

SIMULATIONS_IN_PARALLEL = 2
TOTAL_RUNS = 10

def benchmark_batch_frequency_sweep():
    @filter_jit
    def simulate(params):
        Q, omega_0, alpha, gamma = np.array([10.0, 20.0]), np.array([1.00, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
        gamma[0,0,0,0] = 2.67e-02
        gamma[1,1,1,1] = 5.40e-01
        alpha[0,0,1] = 7.48e-01
        alpha[1,0,0] = 3.74e-01

        MODEL = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
        DRIVING_FREQUENCY = np.linspace(0.5, 1.5, 150)
        MAX_FORCE = 0.3
        DRIVING_AMPLITUDE = np.linspace(0.1 * MAX_FORCE, 1.0 * MAX_FORCE, 10)
        EXCITOR = oscidyn.OneToneExcitation(drive_frequencies=DRIVING_FREQUENCY, drive_amplitudes=DRIVING_AMPLITUDE, modal_forces=np.array([1.0, 1.0]))
        MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
        SOLVER = oscidyn.TimeIntegrationSolver(max_steps=4096*1, n_time_steps=50, verbose=True, throw=False, rtol=1e-3, atol=1e-7)
        SWEEPER = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
        PRECISION = oscidyn.Precision.SINGLE

        frequency_sweep = oscidyn.frequency_sweep(
            model = MODEL,
            sweeper=SWEEPER,
            excitor=EXCITOR,
            solver = SOLVER,
            precision = PRECISION,
            multistarter=MULTISTART,
        )

    dummy_params = np.zeros((SIMULATIONS_IN_PARALLEL,))
    batch_params = jax.tree_util.tree_map(lambda x: np.tile(x, (SIMULATIONS_IN_PARALLEL,) + (1,) * x.ndim), dummy_params)
    batch_sweeps = jax.vmap(simulate)(batch_params)

execution_time = timeit.timeit(number=1, stmt=benchmark_batch_frequency_sweep)
print(f"Benchmark completed in {execution_time:.4f} seconds for {TOTAL_RUNS} runs.")