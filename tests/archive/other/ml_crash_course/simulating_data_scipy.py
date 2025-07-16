import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time

def duffing_rhs_real(Y, t, omega_start, domega_dt, gamma, epsilon, force, omega0):
    # Y = [Re(z), Im(z)]
    z = Y[0] + 1j * Y[1]
    d_omega = omega0 - (omega_start + domega_dt * t)
    dz = (-1j * d_omega - 0.5 * gamma) * z \
         -1j * epsilon * np.abs(z)**2 * z \
         +1j * force
    return [dz.real, dz.imag]

def solve_duffing_scipy(z0, omega0, gamma, epsilon, force,
                        omega_start, omega_stop, t_end, n_time_steps):
    ts = np.linspace(0.0, t_end, n_time_steps)
    domega_dt = (omega_stop - omega_start) / t_end
    omegas = omega_start + domega_dt * ts
    Y0 = [z0.real, z0.imag]
    sol = odeint(duffing_rhs_real, Y0, ts,
                 args=(omega_start, domega_dt, gamma, epsilon, force, omega0))
    z_ts = sol[:,0] + 1j * sol[:,1]
    return z_ts, omegas

def duffing_produce_training_batch_scipy(key, batchsize, ranges,
                                         num_frequency_bins,
                                         force=1.0, omega_start=-4.0,
                                         omega_stop=4.0,
                                         t_end=1000.0, n_time_steps=400):
    # draw random parameters
    rng = np.random.RandomState(key)
    params = [rng.uniform(low, high, size=batchsize)
              for (low, high) in ranges]
    omega0s, gammas, epsilons = params
    # prepare output arrays
    x = np.zeros((batchsize, num_frequency_bins))
    for i in range(batchsize):
        z_ts, omegas = solve_duffing_scipy(
            0.0+0.0j,
            omega0s[i], gammas[i], epsilons[i], force,
            omega_start, omega_stop, t_end, n_time_steps
        )
        amp = np.abs(z_ts)
        # resample onto desired frequency bins
        freq_bins = np.linspace(omega_start, omega_stop, num_frequency_bins)
        x[i] = np.interp(freq_bins, omegas, amp)
    y_target = np.vstack([omega0s, gammas, epsilons]).T
    return x, y_target

# Example usage
if __name__ == "__main__":
    key = 44
    batchsize = 10
    npixels = 200
    ranges = [[-1., 1.], [0.5, 1.5], [0.0, 0.15]]

    start = time.time()
    x, y = duffing_produce_training_batch_scipy(
        key, batchsize, ranges, npixels
    )
    elapsed = time.time() - start
    print(f"Elapsed: {elapsed:.3f}s")

    freq = np.linspace(-4, 4, npixels)
    fig, axes = plt.subplots(1, batchsize, figsize=(10,2), sharey=True)
    for idx, ax in enumerate(axes):
        ax.fill_between(freq, 0, x[idx])
        ax.set_ylim(0, 4)
        ax.axis('off')
    plt.show()
