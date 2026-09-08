from __future__ import annotations
import time
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree
import matplotlib.pyplot as plt

import poscidyn

class DMT(poscidyn.AbstractOscillator):
    def __init__(self, d1: Array = None, d2: Array = None, C1: Array = None, C2: Array = None, C3: Array = None, a0: Array = None):
        self.d1 = d1
        self.d2 = d2
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.a0 = a0

        self.n_modes = 1

    def f_i(self, t: Float, y: Array, args: PyTree, omega_ref: float = 1.0, x_ref: float = 1.0) -> Array:
        eta1, eta2 = y

        def case_true(_): 
            return (-self.d2*eta2) - eta1 - self.C1 - (self.C1/self.a0**2) - self.C2*(self.a0-(1 - eta1))**1.5 - self.C3*(((self.a0-(1 - eta1))**(0.5))*eta2) #Repulsive regime
        def case_false(_):
            return (-self.d1*eta2) - eta1 - self.C1 - (self.C1/(1-eta1)**2)  #Attractive regime

        deta2 = lax.cond(((1.0-eta1) <= self.a0), case_true, case_false, operand=None)

        return jnp.atleast_1d(deta2)

    def t_steady_state(self, driving_frequency: jax.Array, ss_tol: float) -> float:
        t_steady_y = 1000.0
        return t_steady_y

    @property
    def n_dof(self) -> int:
        return self.n_modes

class Intermodulation(poscidyn.AbstractExcitation):
    def __init__(self, B1: Array, y_bar, DOmega: Array, Omegas: Array, lambdas: Array = jnp.array([1.0])):
        omegas = Omegas

        super().__init__(omegas, lambdas)

        self.f_d = B1 * y_bar
        self.DOmega = DOmega

    def f_e(self, t: Float, y: Array, args: PyTree) -> float:
        """Direct external forces of the equations of motion.

        Args:
            t (float): Time
            y (Array): y vector
            args (PyTree): Additional arguments
        """
        f_amp = args.get("f_amp")
        Omega = args["omega"]
        if f_amp is None:
            f_amp = self.f_d * args["lambda"]

        return f_amp * Omega**2 * (jnp.sin((Omega + self.DOmega) * t) + jnp.sin((Omega - self.DOmega) * t))
        
class TestMultistart(poscidyn.AbstractMultistart):
    def __init__(
        self,
        n_init_cond: int = 16,
        max_response_amplitude: float = 1.0,
        random_seed: int | None = 0,
    ):
        super().__init__()
        n_init_cond = int(n_init_cond)
        if n_init_cond < 1:
            raise ValueError("n_init_cond must be >= 1.")

        self.n_init_cond = n_init_cond
        self.max_response_amplitude = max_response_amplitude
        self.random_seed = random_seed

    def generate_simulation_grid(self, model, omegas, f_amps):
        f_amps = jnp.asarray(f_amps)
        if f_amps.ndim != 2:
            raise ValueError(
                f"f_amps must be a 2D array with one modal-force vector per amplitude level, got shape {f_amps.shape}."
            )

        n_modes = model.n_modes
        if f_amps.shape[-1] != n_modes:
            if f_amps.shape[0] == n_modes:
                f_amps = jnp.swapaxes(f_amps, 0, 1)
            else:
                raise ValueError(
                    f"f_amps shape {f_amps.shape} is incompatible with n_modes={n_modes}."
                )

        max_force = jnp.max(f_amps, axis=None)
        max_displacement_per_mode = np.array([self.max_response_amplitude])
        max_velocity_per_mode = np.array([self.max_response_amplitude])

        n_omegas = omegas.shape[0]
        n_f_amps = f_amps.shape[0]
        n_init_cond = self.n_init_cond

        omegas_grid = jnp.tile(omegas[:, None], (1, n_modes))
        f_amps_grid = f_amps

        if self.random_seed is None:
            random_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        else:
            random_seed = int(self.random_seed)

        key = jax.random.PRNGKey(random_seed)
        key_x0, key_v0 = jax.random.split(key)
        unit_x0s_grid = jax.random.uniform(
            key_x0,
            shape=(n_init_cond, n_modes),
            minval=-1.0,
            maxval=1.0,
            dtype=f_amps.dtype,
        )
        unit_v0s_grid = jax.random.uniform(
            key_v0,
            shape=(n_init_cond, n_modes),
            minval=-1.0,
            maxval=1.0,
            dtype=f_amps.dtype,
        )

        x0s_grid = unit_x0s_grid * max_displacement_per_mode[None, :]
        v0s_grid = unit_v0s_grid * max_velocity_per_mode[None, :]

        shape = (n_omegas, n_f_amps, n_init_cond, 1, n_modes)
        omegas_mesh = jnp.broadcast_to(omegas_grid[:, None, None, None, :], shape)
        f_amps_mesh = jnp.broadcast_to(f_amps_grid[None, :, None, None, :], shape)
        x0_mesh = jnp.broadcast_to(x0s_grid[None, None, :, None, :], shape)
        v0_mesh = jnp.broadcast_to(v0s_grid[None, None, :, None, :], shape)

        n_combinations = n_omegas * n_f_amps * n_init_cond
        omegas_mesh = omegas_mesh.reshape(n_combinations, n_modes)
        f_amps_mesh = f_amps_mesh.reshape(n_combinations, n_modes)
        x0_mesh = x0_mesh.reshape(n_combinations, n_modes)
        v0_mesh = v0_mesh.reshape(n_combinations, n_modes)

        return (omegas_mesh, f_amps_mesh, x0_mesh, v0_mesh, shape)
    
OSCILLATOR = DMT(d1 = 0.001219, d2 = 0.001219, C1 = -1.6935e-8 * 2, C2 = 0.5630, C3 = 0.1798, a0 = 0.002 * 2)
SYNTHETIC_SWEEP = poscidyn.synthetic_sweep.NearestNeighbour(sweep_direction=[poscidyn.synthetic_sweep.Forward(), poscidyn.synthetic_sweep.Backward()])
MULTISTART = TestMultistart(n_init_cond=16, max_response_amplitude=1.0)
SOLVER = poscidyn.solver.TimeIntegration(
    multistart=MULTISTART, synthetic_sweep=SYNTHETIC_SWEEP,
    max_steps=4096 * 20, n_time_steps=100, verbose=True, throw=False, rtol=1e-9, atol=1e-12,
    periods_to_retain=100
)
RESPONSE_MEASURE = poscidyn.Demodulation(multiples=(1,), modal_contributions=np.array([1.0]), window=None)
PRECISION = poscidyn.Precision.DOUBLE

time_response = poscidyn.time_response(
    oscillator=OSCILLATOR,
    excitation= Intermodulation(B1 = 1.56598, y_bar = 0.00, DOmega = 0.0012195, Omegas = np.array([0.1])),
    initial_displacement=np.array([0.0]),
    initial_velocity=np.array([0.0]),
    solver=SOLVER,
    precision=PRECISION,
    only_save_steady_state=True
)

plt.figure(figsize=(8, 5))
ts, xs, vs = time_response
plt.plot(ts, xs[:, 0], label="Displacement")
plt.plot(ts, vs[:, 0], label="Velocity")
plt.xlabel("Time")
plt.ylabel("Response")
plt.title("Time Response")
plt.grid()
plt.legend()
plt.show()


# start_time = time.time()

# frequency_sweep = poscidyn.frequency_sweep(
#     oscillator=OSCILLATOR,
#     excitation= Intermodulation(B1 = 1.56598, y_bar = 0.00505, DOmega = 0.0012195, Omegas = np.linspace(0.1, 2.0, 512)),
#     solver=SOLVER,
#     response_measure=RESPONSE_MEASURE,
#     precision=PRECISION,
# )

# end_time = time.time()
# print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")
# n_successful = frequency_sweep.stats["n_successful"]
# n_total = frequency_sweep.stats["n_total"]
# success_rate = frequency_sweep.stats["success_rate"]
# print(
#     f"Successful periodic solutions: {n_successful}/{n_total} "
#     f"({success_rate:.1%})"
# )


# sweep =  frequency_sweep.modal_superposition.amplitudes['forward']

# plt.figure(figsize=(8, 5))
# plt.plot(EXCITATION.omegas, sweep, marker='o')
# plt.xlabel("Driving Frequency")
# plt.ylabel("Response Amplitude")
# plt.title("Frequency Response")
# plt.grid()
# plt.show()  
