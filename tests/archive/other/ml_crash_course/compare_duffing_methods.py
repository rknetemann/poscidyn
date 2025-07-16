import time
import os

os.environ["JAX_PLATFORMS"] = "cpu"  # Force JAX to use the CPU backend

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import diffrax
import timeit
from simulating_data import solve_duffing          # JAX version
from simulating_data_scipy import solve_duffing_scipy  # SciPy version

def solve_duffing_diffrax(z0, omega0, gamma, epsilon, force,
                          omega_start, omega_stop, t_end, n_time_steps):
    ts = jnp.linspace(0.0, t_end, n_time_steps)
    domega_dt = (omega_stop - omega_start) / t_end
    params = (omega_start, domega_dt, gamma, epsilon, force, omega0)
    def vf(t, y, args):
        omega_start_, domega_dt_, gamma_, eps_, force_, omega0_ = args
        z = y[0] + 1j * y[1]
        d_omega = omega0_ - (omega_start_ + domega_dt_ * t)
        dz = ((-1j * d_omega - 0.5 * gamma_) * z
              -1j * eps_ * jnp.abs(z)**2 * z
              +1j * force_)
        return jnp.stack([jnp.real(dz), jnp.imag(dz)])
    term = diffrax.ODETerm(vf)
    solver = diffrax.Dopri8()
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        term, solver,
        t0=0.0, t1=t_end,
        dt0=(t_end / n_time_steps),
        y0=jnp.array([z0.real, z0.imag]),
        args=params,
        saveat=saveat,
    )
    y = sol.ys
    z_ts = y[:,0] + 1j * y[:,1]
    omegas = omega_start + domega_dt * ts
    return z_ts, omegas

def compare_single(z0, omega0, gamma, epsilon, force,
                   omega_start, omega_stop, t_end, n_time_steps,
                   repeats=5):
    # JAX solve timing
    def _run_jax():
        zs, *_ = solve_duffing(
            z0, omega0, gamma, epsilon,
            force, omega_start, omega_stop,
            t_end, n_time_steps
        )
        zs.block_until_ready()
    dt_jax = timeit.timeit(_run_jax, number=repeats) / repeats
    zs_jax, ts_jax, omegas_jax = solve_duffing(
        z0, omega0, gamma, epsilon,
        force, omega_start, omega_stop,
        t_end, n_time_steps
    )
    
    # SciPy solve timing
    def _run_sci():
        solve_duffing_scipy(
            z0, omega0, gamma, epsilon,
            force, omega_start, omega_stop,
            t_end, n_time_steps
        )
    dt_sci = timeit.timeit(_run_sci, number=repeats) / repeats
    zs_sci, omegas_sci = solve_duffing_scipy(
        z0, omega0, gamma, epsilon,
        force, omega_start, omega_stop,
        t_end, n_time_steps
    )
    
    # Diffrax solve timing
    def _run_dx():
        solve_duffing_diffrax(
            z0, omega0, gamma, epsilon,
            force, omega_start, omega_stop,
            t_end, n_time_steps
        )
    dt_dx = timeit.timeit(_run_dx, number=repeats) / repeats
    zs_dx, omegas_dx = solve_duffing_diffrax(
        z0, omega0, gamma, epsilon,
        force, omega_start, omega_stop,
        t_end, n_time_steps
    )

    return (omegas_jax, np.abs(np.array(zs_jax)), dt_jax), \
           (omegas_sci, np.abs(zs_sci), dt_sci), \
           (omegas_dx, np.abs(zs_dx), dt_dx)

if __name__ == "__main__":
    # parameters
    z0 = 0.0+0.0j
    omega0, gamma, epsilon = -1.0, 1.0, 0.1
    force = 1.0
    omega_start, omega_stop = -4.0, 4.0
    t_end, n_time_steps = 1000.0, 2000

    (om_j, amp_j, t_j), (om_s, amp_s, t_s), (om_d, amp_d, t_d) = compare_single(
        z0, omega0, gamma, epsilon, force,
        omega_start, omega_stop, t_end, n_time_steps,
        repeats=5
    )

    print(f"JAX time:   {t_j:.3f}s")
    print(f"SciPy time: {t_s:.3f}s")
    print(f"Diffrax time: {t_d:.3f}s")

    # plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(12,4), sharey=True)
    axes[0].plot(om_j, amp_j, color='C0'); axes[0].set_title("JAX RK4")
    axes[1].plot(om_s, amp_s, color='C1'); axes[1].set_title("SciPy odeint")
    axes[2].plot(om_d, amp_d, color='C2'); axes[2].set_title("Diffrax Tsit5")
    for ax in axes:
        ax.set_xlabel("drive freq")
    axes[0].set_ylabel("|z|")
    plt.tight_layout()
    plt.show()
