import numpy as np
import poscidyn
import timeit

TOTAL_RUNS = 10

def benchmark_single_frequency_sweep():
    Q, omega_0, alpha, gamma = np.array([10.0, 20.0]), np.array([1.00, 2.0]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
    gamma[0,0,0,0] = 2.67e-02
    gamma[1,1,1,1] = 5.40e-01
    alpha[0,0,1] = 7.48e-01
    alpha[1,0,0] = 3.74e-01

    MODEL = poscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
    DRIVING_FREQUENCY = np.linspace(0.5, 1.5, 150)
    MAX_FORCE = 0.3
    DRIVING_AMPLITUDE = np.linspace(0.1 * MAX_FORCE, 1.0 * MAX_FORCE, 10)
    EXCITOR = poscidyn.OneToneExcitation(drive_frequencies=DRIVING_FREQUENCY, drive_amplitudes=DRIVING_AMPLITUDE, modal_forces=np.array([1.0, 1.0]))
    MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(3, 3), linear_response_factor=1.0)
    SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*1, n_time_steps=50, verbose=True, throw=False, rtol=1e-3, atol=1e-7)
    SWEEPER = poscidyn.NearestNeighbourSweep(sweep_direction=[poscidyn.Forward(), poscidyn.Backward()])
    PRECISION = poscidyn.Precision.SINGLE

    frequency_sweep = poscidyn.frequency_sweep(
        model = MODEL,
        sweeper=SWEEPER,
        excitor=EXCITOR,
        solver = SOLVER,
        precision = PRECISION,
        multistarter=MULTISTART,
    )

execution_time = timeit.timeit(number=10, stmt=benchmark_single_frequency_sweep)
print(f"Benchmark completed in {execution_time:.4f} seconds for 10 runs.")
