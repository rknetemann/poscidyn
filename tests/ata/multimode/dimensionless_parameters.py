import numpy as np
import poscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


### Circular drum ###
# Scaling parameters (Supplementary Table 2)

def netto_scalilng(E, h, R, x_ref, omega_0_ref, rho):
    return E / (R**4 * rho * np.pi) * x_ref**2 / omega_0_ref**2

# Parameter ranges based on common materials and geometries

R_range = [1.0e-6, 20.0e-6]  # ms
h_range = [1.0e-9, 20.0e-9]  # m
E_range = [100e9, 1.0e12]  # Pa
rho_range = [2200.0, 2300.0]  # kg/m
T0_range = [0.1, 1.0]  # N/m
x_ref_range = [1.0e-9, 100.0e-9]  # m
omega_0_ref_range = [1.0e6, 100.0e6]  # Hz

# Non-dimensional gamma values for 2DOF system from Supplementary Table 3 and 4

mass0_table = 0.204
mass1_table = 0.195

gamma0000_table = 2.55
gamma1111_table = 18.7
gamma0011_table = 8.61
gamma1001_table = 8.57

# Deriving nondimensional gamma ranges using geometric/material and displacement/frequency scalings

min_netto_scaling = netto_scalilng(E_range[0], h_range[0], R_range[1], x_ref_range[0], omega_0_ref_range[1], rho_range[1])
max_netto_scaling = netto_scalilng(E_range[1], h_range[1], R_range[0], x_ref_range[1], omega_0_ref_range[0], rho_range[0])
print(f"Netto scaling range: {min_netto_scaling:.2e} to {max_netto_scaling:.2e} 1/(m^2 s^2 kg)")

netto_scaling = np.logspace(np.log10(min_netto_scaling), np.log10(max_netto_scaling), num=10)
print("Netto scaling values:", netto_scaling)

gamma0000 = gamma0000_table / mass0_table * netto_scaling
gamma1111 = gamma1111_table / mass1_table * netto_scaling
gamma0011 = gamma0011_table / mass0_table * netto_scaling
gamma1001 = gamma1001_table / mass1_table * netto_scaling
# print("Gamma 0000 values:", gamma0000)
# print("Gamma 1111 values:", gamma1111)
# print("Gamma 0011 values:", gamma0011)
# print("Gamma 1001 values:", gamma1001)

# print(f"Gamma 0000 range: {gamma0000.min():.2e} to {gamma0000.max():.2e}")
# print(f"Gamma 1111 range: {gamma1111.min():.2e} to {gamma1111.max():.2e}")
# print(f"Gamma 0011 range: {gamma0011.min():.2e} to {gamma0011.max():.2e}")
# print(f"Gamma 1001 range: {gamma1001.min():.2e} to {gamma1001.max():.2e}")

def F (c, omega_0, Q, gamma):
    omega_max = np.sqrt(omega_0**2 + c * omega_0**2)
    return np.sqrt(((omega_max**2 * omega_0**2) / Q**2) * ((omega_max**2 - omega_0**2)**2 / (3/4 * gamma)))

def F_alt (c, omega_0, Q, gamma):
    return np.sqrt(4 * omega_0**6 / (3 * gamma * Q**2) * (c + c**2))

def F_alt2 (c, omega_0, Q, gamma):
    return np.sqrt(4 * omega_0**6 / (3 * gamma * Q**2) * (c + 1 / (2*Q**2)) * (1 + c + 1 / (4 * Q **2)))

###### FREQUENCY SWEEPING ########

import numpy as np
import poscidyn
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

index = 5
Q, omega_0, alpha, gamma = np.array([50.0, 50.0]), np.array([1.0, 1.73]), np.zeros((2,2,2)), np.zeros((2,2,2,2))
gamma[0,0,0,0] = gamma0000[index] 
gamma[0,0,1,1] = gamma0011[index] 
gamma[1,1,1,1] = gamma1111[index] 
gamma[1,0,0,1] = gamma1001[index]
modal_forces = np.array([1.0, 1.0])

f0000 = F(0.5, omega_0[0], Q[0], gamma0000[index])
f0000_alt = F_alt(0.5, omega_0[0], Q[0], gamma0000[index])
f0000_alt2 = F_alt2(0.5, omega_0[0], Q[0], gamma0000[index])

print(f"Estimated force amplitude for c=0.5: {f0000:.2e} (alt: {f0000_alt:.2e}, alt2: {f0000_alt2:.2e})")

driving_frequency = np.linspace(0.75, 1.25, 201)
driving_amplitude = np.linspace(0.1, 1.0, 10) * f0000_alt2
# driving_amplitude = np.array([1.0]) * f0000_alt2
modal_forces = np.array([1.0])

MODEL = poscidyn.NonlinearOscillator(Q=Q, alpha=alpha, gamma=gamma, omega_0=omega_0)
EXCITOR = poscidyn.OneToneExcitation(driving_frequency, driving_amplitude, modal_forces)
MULTISTART = poscidyn.LinearResponseMultistart(init_cond_shape=(11, 11), linear_response_factor=1.0)
SOLVER = poscidyn.TimeIntegrationSolver(max_steps=4096*1, n_time_steps=50, verbose=True, throw=False, rtol=1e-5, atol=1e-7)
SWEEPER = poscidyn.NearestNeighbourSweep(sweep_direction=[poscidyn.Forward(), poscidyn.Backward()])
PRECISION = poscidyn.Precision.SINGLE

def _extract_gamma_diagonal(gamma: np.ndarray):
    arr = np.asarray(gamma)
    if arr.ndim < 4:
        return arr.ravel().tolist()[: arr.size]

    diag = []
    max_modes = min(2, arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])
    for i in range(max_modes):
        diag.append(float(arr[i, i, i, i]))
    return diag


def _extract_gamma_entries(gamma: np.ndarray, max_entries: int = 3):
    arr = np.asarray(gamma)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _extract_alpha_entries(alpha: np.ndarray, max_entries: int = 3):
    arr = np.asarray(alpha)
    non_zero = np.argwhere(arr != 0)
    entries = []
    for idx in non_zero[:max_entries]:
        tuple_idx = tuple(idx.tolist())
        entries.append((tuple_idx, float(arr[tuple_idx])))
    return entries, int(non_zero.shape[0])


def _format_param_text(Q: np.ndarray, omega_0: np.ndarray, alpha: np.ndarray, gamma: np.ndarray, modal_forces: np.ndarray) -> str:
    q_vals = np.asarray(Q).ravel()
    omega_vals = np.asarray(omega_0).ravel()
    alpha_entries, alpha_total = _extract_alpha_entries(alpha)
    gamma_entries, gamma_total = _extract_gamma_entries(gamma)

    parts = []
    if q_vals.size:
        parts.append(f"Q=[{', '.join(f'{val:.2f}' for val in q_vals[:2])}]")
    if omega_vals.size:
        parts.append(f"omega0=[{', '.join(f'{val:.2f}' for val in omega_vals[:2])}]")
    if alpha_entries:
        formatted_alpha = ", ".join(f"{idx}={val:.2e}" for idx, val in alpha_entries)
        if alpha_total > len(alpha_entries):
            formatted_alpha += ", ..."
        parts.append(f"alpha={formatted_alpha}")
    if gamma_entries:
        formatted_gamma = ", ".join(f"{idx}={val:.2e}" for idx, val in gamma_entries)
        if gamma_total > len(gamma_entries):
            formatted_gamma += ", ..."
        parts.append(f"gamma={formatted_gamma}")
    if modal_forces.size:
        parts.append(f"modal_forces=[{', '.join(f'{val:.2f}' for val in modal_forces[:2])}]")
    return "\n".join(parts)


def plot_sweep(ax, drive_freqs, drive_amps, sweeped_solutions, param_text: str, use_responsivity: bool = True) -> None:
    forward = sweeped_solutions.get("forward")
    backward = sweeped_solutions.get("backward")

    if forward is None and backward is None:
        raise ValueError("No sweeped solutions to plot.")
    
    max_forward = np.max(forward)
    max_backward = np.max(backward)

    max_value = max(
        val for val in [max_forward, max_backward] if val is not None
    )
    print(f"Max value across forward and backward: {max_value}")

    arg_max_forward = np.unravel_index(np.argmax(forward), forward.shape) if forward is not None else None
    arg_max_backward = np.unravel_index(np.argmax(backward), backward.shape) if backward is not None else None

    frequency_at_max_forward = drive_freqs[arg_max_forward[0]] if arg_max_forward is not None else None
    if frequency_at_max_forward is not None and isinstance(frequency_at_max_forward, np.ndarray):
        frequency_at_max_forward = float(frequency_at_max_forward)
    print(f"Max forward at frequency: {frequency_at_max_forward}")

    # Store original displacement before responsivity transformation
    drive_amps_array = np.asarray(drive_amps)
    original_forward = forward.copy() if forward is not None else None
    original_backward = backward.copy() if backward is not None else None

    # Calculate nondimensional parameters for each force amplitude using original displacement
    gamma_ndim_per_force = []
    force_ndim_per_force = []
    
    for idx in range(drive_amps_array.size):
        # Get max displacement for this specific force amplitude (from original, not responsivity)
        max_disp_forward = np.max(original_forward[:, idx]) if original_forward is not None else 0
        max_disp_backward = np.max(original_backward[:, idx]) if original_backward is not None else 0
        max_disp_this_force = max(max_disp_forward, max_disp_backward)
        
        # Calculate nondimensional gamma: gamma_ndim = x_max^2 * gamma
        gamma_ndim = max_disp_this_force**2 * gamma[0,0,0,0]
        gamma_ndim_per_force.append(gamma_ndim)
        
        # Calculate nondimensional force: force_ndim = force / x_max
        force_ndim = drive_amps_array[idx] / max_disp_this_force if max_disp_this_force > 0 else 0
        force_ndim_per_force.append(force_ndim)
        
        print(f"Force {idx}: F={drive_amps_array[idx]:.3e}, x_max={max_disp_this_force:.3e}, "
              f"gamma_ndim={gamma_ndim:.3e}, force_ndim={force_ndim:.3e}")

    # Calculate responsivity (displacement / driving force amplitude) if enabled
    if use_responsivity:
        if forward is not None:
            forward = forward / drive_amps_array[np.newaxis, :]
        if backward is not None:
            backward = backward / drive_amps_array[np.newaxis, :]
        ylabel = "Responsivity (Displacement / Force)"
        title_suffix = "(Responsivity)"
    else:
        ylabel = "Max displacement"
        title_suffix = ""


    drive_freqs = np.asarray(drive_freqs)
    colors = plt.cm.viridis(np.linspace(0, 1, drive_amps_array.size))

    for idx, (amp, color) in enumerate(zip(drive_amps_array, colors)):
        if forward is not None:
            ax.plot(
                drive_freqs,
                forward[:, idx],
                color=color,
                linestyle="-",
                linewidth=1.3,
            )
        if backward is not None:
            ax.plot(
                drive_freqs,
                backward[:, idx],
                color=color,
                linestyle="--",
                linewidth=1.1,
            )

    ax.set_title(
        f"Frequency sweep {title_suffix}\nMax displacement: {max_value:.2f} (omega: {frequency_at_max_forward:.2f})"
    )
    ax.set_xlabel("Drive frequency")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

    if param_text:
        ax.text(
            0.02,
            0.98,
            param_text,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65),
        )

    amp_handles = [
        Line2D([0], [0], color=color, lw=1.8) for color in colors
    ]
    amp_labels = [f"F={amp:.3f}" for amp in drive_amps_array]
    legend1 = ax.legend(
        amp_handles,
        amp_labels,
        title="Drive amplitude",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=7,
        frameon=False,
        ncol=2,
    )

    if forward is not None and backward is not None:
        style_handles = [
            Line2D([0], [0], color="k", linestyle="-", lw=1.5),
            Line2D([0], [0], color="k", linestyle="--", lw=1.5),
        ]
        style_labels = ["Forward sweep", "Backward sweep"]
        legend2 = ax.legend(
            style_handles,
            style_labels,
            loc="lower right",
            fontsize=7,
            frameon=False,
        )
        ax.add_artist(legend1)
    else:
        ax.add_artist(legend1)

start_time = time.time()

frequency_sweep = poscidyn.frequency_sweep(
    model = MODEL,
    sweeper=SWEEPER,
    excitor=EXCITOR,
    solver = SOLVER,
    precision = PRECISION,
    multistarter=MULTISTART,
) #n_freq, n_amp, n_init_disp, n_init_vel

end_time = time.time()
print(f"Frequency sweep completed in {end_time - start_time:.2f} seconds.")
print(
    f"Successful periodic solutions: {frequency_sweep.n_successful}/{frequency_sweep.n_total} "
    f"({frequency_sweep.success_rate:.1%})"
)


fig, ax = plt.subplots(figsize=(10, 6))
plot_sweep(
    ax=ax,
    drive_freqs=EXCITOR.drive_frequencies,
    drive_amps=EXCITOR.drive_amplitudes,
    sweeped_solutions=frequency_sweep.sweeped_periodic_solutions,
    param_text=_format_param_text(Q, omega_0, alpha, gamma, EXCITOR.modal_forces),
    use_responsivity=True,  # Set to False to plot raw displacement
)
plt.tight_layout()
plt.show()

