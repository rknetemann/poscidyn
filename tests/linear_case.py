import numpy as np
import h5py
import matplotlib.pyplot as plt

HDF5_FILE = "/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/converted/batch_0_2025-12-18_17-32-41_converted.hdf5"
HDF5_FILE = "/home/raymo/Projects/oscidyn/data/simulations/18_12_2025/batch_0_2025-12-18_17-32-41_converted.hdf5"

def _normalize_f_amp(f_amp: np.ndarray) -> np.ndarray:
    """Ensure f_amp is (n_modes, n_amp_sets)."""
    f_amp = np.asarray(f_amp)
    if f_amp.ndim == 1:
        return f_amp[:, np.newaxis]
    if f_amp.ndim == 2:
        return f_amp
    raise ValueError(f"f_amp has unsupported shape {f_amp.shape}; expected 1D or 2D.")

def linear_response_complex(Q: np.ndarray,
                            omega_0: np.ndarray,
                            f_omega: np.ndarray,
                            f_amp: np.ndarray) -> np.ndarray:
    """
    Returns complex total response X(omega) for each frequency and amplitude set.
    Shapes:
      Q, omega_0: (n_modes,)
      f_omega:    (n_freq,)
      f_amp:      (n_modes,) or (n_modes, n_amp_sets)
    Output:
      X_total:    (n_freq, n_amp_sets) complex
    """
    f_amp = _normalize_f_amp(f_amp)

    # Broadcast to (n_modes, n_freq, n_amp_sets)
    Q = Q[:, np.newaxis, np.newaxis]
    omega_0 = omega_0[:, np.newaxis, np.newaxis]
    w = f_omega[np.newaxis, :, np.newaxis]
    f_amp = f_amp[:, np.newaxis, :]

    denom = (omega_0**2 - w**2) + 1j * (w * omega_0 / Q)
    X_modes = f_amp / denom              # complex modal contributions
    X_total = np.sum(X_modes, axis=0)    # complex superposition over modes
    return X_total

def linear_response_amplitude(Q, omega_0, f_omega, f_amp):
    X_total = linear_response_complex(Q, omega_0, f_omega, f_amp)
    return np.abs(X_total)


def extract_sweep_series(sweep_data: np.ndarray, f_omega: np.ndarray) -> np.ndarray:
    sweep_arr = np.asarray(sweep_data)
    if sweep_arr.ndim >= 2 and sweep_arr.shape[0] != f_omega.shape[0] and sweep_arr.shape[1] == f_omega.shape[0]:
        sweep_arr = sweep_arr.T
    if sweep_arr.shape[0] != f_omega.shape[0]:
        raise ValueError(f"sweep_data shape {sweep_arr.shape} incompatible with f_omega length {f_omega.shape[0]}")
    if sweep_arr.ndim == 1:
        return sweep_arr
    return np.linalg.norm(sweep_arr.reshape(sweep_arr.shape[0], -1), axis=1)

with h5py.File(HDF5_FILE, "r") as f:
    forward_sweeps = f["forward_sweeps"]
    sim_names = list(forward_sweeps.keys())
    
    for sim_name in sim_names:
        sim_data = forward_sweeps[sim_name]

        sweep_data = np.array(sim_data)

        Q = np.array(sim_data.attrs["Q"])
        omega_0 = np.array(sim_data.attrs["scaled_omega_0"])
        f_omega = np.array(sim_data.attrs["scaled_f_omegas"])
        f_amp = _normalize_f_amp(np.array(sim_data.attrs["scaled_f_amp"]))
        # print(f_amp.shape)  # optional debug
        X_total = linear_response_complex(Q, omega_0, f_omega, f_amp)
        response = np.abs(X_total)
        sweep_series = extract_sweep_series(sweep_data, f_omega)

        eps = 1e-12

        difference = sweep_series[:, np.newaxis] - response
        rel_l2 = np.linalg.norm(difference, axis=0) / (np.linalg.norm(sweep_series) + eps)

        print(f"Sim: {sim_name} | Relative L2 error per amp-set: {rel_l2}")

        if rel_l2 < 0.001:
            plt.figure(figsize=(8, 5))
            plt.plot(f_omega, sweep_series, label="Sim data", linestyle="--", color="k")
            for amp_idx in range(response.shape[1]):
                amp_set = f_amp[:, amp_idx]
                label = f"Amp set {amp_idx} | ||F||={np.linalg.norm(amp_set):.3g}"
                plt.plot(f_omega, response[:, amp_idx], label=label)
            plt.title(f"Linear Response for {sim_name}")
            plt.xlabel("Scaled Frequency (f_omega)")
            plt.ylabel("Response Amplitude")
            plt.legend()
            plt.grid()
            plt.show()

