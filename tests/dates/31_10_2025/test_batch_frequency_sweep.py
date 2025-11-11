import numpy as np
import jax.numpy as jnp
import jax
import time
import matplotlib.pyplot as plt

import oscidyn

# --------------------------
# Config
# --------------------------
N_DUFFING_IN_PARALLEL = 2
DUFFING_COEFFICIENTS = np.array([0.01, 0.02, 0.03, 0.04])

SWEEP = oscidyn.NearestNeighbourSweep(sweep_direction=[oscidyn.Forward(), oscidyn.Backward()])
EXCITATION = oscidyn.OneToneExcitation(drive_frequencies=np.linspace(0.1, 3.0, 151), drive_amplitudes=np.linspace(0.01, 0.1, 5), 
                                       modal_forces=np.array([1.0, 0.5]))
MULTISTART = oscidyn.LinearResponseMultistart(init_cond_shape=(5, 5), linear_response_factor=1.2)
SOLVER = oscidyn.TimeIntegrationSolver(n_time_steps=200, max_steps=4096*3, verbose=True, throw=False, rtol=1e-4, atol=1e-7)
PRECISION = oscidyn.Precision.SINGLE

start_time = time.time()

@jax.jit
def batched_frequency_sweep(duffing: jax.Array):
    Q = jnp.array([50.0, 23.0])
    omega_0 = jnp.array([1.0, 2.0])
    alpha = jnp.zeros((2, 2, 2))
    gamma = jnp.zeros((2, 2, 2, 2))
    gamma = gamma.at[0, 0, 0, 0].set(duffing)

    model = oscidyn.BaseDuffingOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)

    return oscidyn.frequency_sweep(
        model=model,
        sweeper=SWEEP,
        excitor=EXCITATION,
        solver=SOLVER,
        multistarter=MULTISTART,
        precision=PRECISION,
    )

# --------------------------
# Run in batches and collect results
# --------------------------
n_d = DUFFING_COEFFICIENTS.shape[0]
n_batches = (n_d + N_DUFFING_IN_PARALLEL - 1) // N_DUFFING_IN_PARALLEL  # Ceiling division

results = []  # list of dicts: {"duffing": float, "sweep": sweep_dict}

for i in range(n_batches):
    start_idx = i * N_DUFFING_IN_PARALLEL
    end_idx = min(start_idx + N_DUFFING_IN_PARALLEL, n_d)

    batch_duffing = DUFFING_COEFFICIENTS[start_idx:end_idx]
    # vmap over duffing values in this batch
    batch_sweeps = jax.vmap(batched_frequency_sweep)(batch_duffing)

    # batch_sweeps is a pytree with leading batch dimension
    # Extract each element of the batch and stash with its duffing value
    for j, dval in enumerate(batch_duffing):
        # Slice out the j-th element for every array in the pytree
        single = {k: (np.array(v[j]) if isinstance(v, (np.ndarray, jnp.ndarray)) else v)
                  for k, v in batch_sweeps.items()}
        results.append({"duffing": float(dval), "sweep": single})

end_time = time.time()
elapsed = end_time - start_time
simulations_per_second = n_d / elapsed
print(f"Time taken: {elapsed:.2f} seconds")
print(f"Simulations per second: {simulations_per_second:.2f}")

# --------------------------
# Plotting (like graphene_resonator.py) – one figure per Duffing coefficient
# --------------------------
for item in results:
    duffing_val = item["duffing"]
    sweep = item["sweep"]

    max_x_total = np.array(sweep["max_x_total"])  # (n_freq, n_amp, n_init_disp, n_init_vel)
    max_x_modes = np.array(sweep["max_x_modes"])  # (n_freq, n_amp, n_init_disp, n_init_vel, n_modes)

    # Dimensions
    n_freq, n_amp, n_init_disp, n_init_vel = max_x_total.shape
    n_modes = max_x_modes.shape[-1]

    # Build scatter arrays as in graphene_resonator.py
    frequencies = []
    responses_total = []
    colors = []

    for i_disp in range(n_init_disp):
        for i_vel in range(n_init_vel):
            for i_amp in range(n_amp):
                frequencies.extend(EXCITATION.drive_frequencies.tolist())
                responses_total.extend(max_x_total[:, i_amp, i_disp, i_vel].tolist())
                # color by the first drive component amplitude (same convention as example)
                colors.extend([EXCITATION.drive_amplitudes[i_amp]] * n_freq)

    plt.figure(figsize=(12, 8))

    # Total response
    plt.subplot(n_modes + 1, 1, 1)
    scatter = plt.scatter(frequencies, responses_total, c=colors, cmap='viridis', alpha=0.7)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Driving Amplitude")
    plt.title(f"Total Response - BaseDuffing (γ₀₀₀₀={duffing_val:.3f})")
    plt.xlabel("Driving Frequency")
    plt.ylabel("Response Amplitude")
    plt.grid(True)

        # --- Plot sweep lines: exactly n_drive_amplitudes per direction ---
    sweeped_frequencies = EXCITATION.drive_frequencies
    sweeps = sweep["sweeped_periodic_solutions"]

    def to_total(arr):
        """
        Convert sweep array to total response of shape (n_freq, n_amp).
        - (n_freq, n_amp): already total -> return as is
        - (n_modes, n_freq, n_amp): collapse modes via Euclidean norm
        """
        if arr is None:
            return None
        if arr.ndim == 2:
            # (n_freq, n_amp)
            return arr
        if arr.ndim == 3:
            # (n_modes, n_freq, n_amp) -> total over modes
            return np.linalg.norm(arr, axis=0)
        raise ValueError(f"Unexpected sweep array shape: {arr.shape}")

    fwd = to_total(sweeps.get("forward", None))
    bwd = to_total(sweeps.get("backward", None))

    # Draw one line per drive amplitude (NOT per mode)
    if fwd is not None:
        for amp_idx, amp in enumerate(EXCITATION.drive_amplitudes):
            y = fwd[:, amp_idx]
            plt.plot(sweeped_frequencies, y, "r-", lw=1.5,
                     label="Forward" if amp_idx == 0 else None)

    if bwd is not None:
        for amp_idx, amp in enumerate(EXCITATION.drive_amplitudes):
            y = bwd[:, amp_idx]
            plt.plot(sweeped_frequencies, y, "b--", lw=1.5,
                     label="Backward" if amp_idx == 0 else None)

    plt.legend(ncol=2, fontsize=8, frameon=False)


    # Per-mode responses
    for mode in range(n_modes):
        mode_responses = []
        for i_disp in range(n_init_disp):
            for i_vel in range(n_init_vel):
                for i_amp in range(n_amp):
                    mode_responses.extend(max_x_modes[:, i_amp, i_disp, i_vel, mode].tolist())

        plt.subplot(n_modes + 1, 1, mode + 2)
        scatter = plt.scatter(frequencies, mode_responses, c=colors, cmap='viridis', alpha=0.7)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Driving Amplitude")
        plt.title(f"Mode {mode + 1} Response")
        plt.xlabel("Driving Frequency")
        plt.ylabel("Response Amplitude")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
