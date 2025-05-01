# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401

from modal_eom_improved import Model as ModalEOM

# ────────────── switches ────────────────────────────────────
RUN_TIME   = False     # single-tone time trace
RUN_FREQ   = True      # frequency-response curve
RUN_FORCE  = False     # force-sweep surface
PLOT_DIMLESS = True   # False→ physical units, True → dimensionless

# ────────────── build & scale model ─────────────────────────
N   = 4
mdl = ModalEOM.from_example(N).non_dimensionalise()    # returns itself
T0, Q0 = mdl.T0, mdl.Q0

# helpers ----------------------------------------------------
to_hat  = lambda x, unit: x * unit       # phys → hat
to_phys = lambda x, unit: x / unit       # hat → phys

# eigen-frequencies ------------------------------------------
eig_w_hat  = mdl.eigenfrequencies()          # rad / ŝ
eig_f_hat  = eig_w_hat / (2*np.pi)           # dimensionless “Hz”
eig_f_Hz   = eig_f_hat / T0                  # physical Hz

# =============== study 1: time response =====================
if RUN_TIME:
    print("\nCalculating time response …")
    y0_hat      = jnp.zeros(2*N)
    t_end_hat   = to_hat(200.0, 1/T0)                # 200 s
    w_drive_hat = to_hat(0.49*2*np.pi, T0)           # rad/s ⋅ T0

    ts_h, qs_h, _ = mdl.time_response(
        y0=y0_hat, t_end=t_end_hat, n_steps=4000,
        f_omega=jnp.array([w_drive_hat])
    )

    # choose plotting units ---------------------------------
    if PLOT_DIMLESS:
        t_plot = ts_h[0]
        q_plot = qs_h[0]                 # shape (time, mode)
        xlabel = r"dimensionless time $\hat t$"
    else:
        t_plot = to_phys(ts_h[0], 1/T0)
        q_plot = to_phys(qs_h[0], 1/Q0)
        xlabel = "Time [s]"

    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(t_plot, q_plot[:, m], label=f"Mode {m+1}")
    plt.xlabel(xlabel);  plt.ylabel("Amplitude")
    plt.title("Time response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# =============== study 2: frequency sweep ===================
if RUN_FREQ:
    print("\nCalculating frequency response …")
    w_min, w_max, n_w = 0.0, 1*2*np.pi, 100        # phys rad/s
    w_sweep_hat = jnp.linspace(w_min, w_max, n_w) * T0

    y0_hat, t_end_hat = jnp.zeros(2*N), to_hat(100.0, 1/T0)
    _, q_st_hat, _ = mdl.frequency_response(
        y0=y0_hat, t_end=t_end_hat, n_steps=2000,
        discard_frac=0.8, f_omega=w_sweep_hat
    )

    # units for x-axis and amplitude ------------------------
    if PLOT_DIMLESS:
        freq = w_sweep_hat / (2*np.pi)           # dimensionless “Hz”
        eig_lines = eig_f_hat
        amps = q_st_hat
        xlabel = r"Drive frequency $\hat f_d$"
        amp_label = r"steady $|\hat q|$"
    else:
        freq = w_sweep_hat / (2*np.pi*T0)        # physical Hz
        eig_lines = eig_f_Hz
        amps = q_st_hat * Q0
        xlabel = "Drive frequency [Hz]"
        amp_label = r"steady $|q|$"

    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(freq, amps[:, m], label=f"Mode {m+1}")
    for f in eig_lines:
        plt.axvline(f, ls="--", color="r", alpha=.6)
    plt.xlabel(xlabel); plt.ylabel(amp_label)
    plt.title("Frequency response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()

# =============== study 3: force sweep =======================
if RUN_FORCE:
    print("\nCalculating force sweep …")
    f_amps_phys = np.array([0, 1, 5, 10, 15, 16.5, 16.9])
    f_amps_hat  = f_amps_phys * T0**2 / Q0

    w_sweep_hat = jnp.linspace(0, 1*2*np.pi, 400) * T0
    y0_hat, t_end_hat = jnp.zeros(2*N), to_hat(200.0, 1/T0)

    _, resp_hat = mdl.force_sweep(
        y0=y0_hat, t_end=t_end_hat, n_steps=4000,
        discard_frac=0.8, f_amp=f_amps_hat, f_omega=w_sweep_hat
    )                                   # shape (n_F, n_w, N)

    if PLOT_DIMLESS:
        freq = w_sweep_hat / (2*np.pi)
        amps = resp_hat
        xlabel = r"$\hat f_d$"
    else:
        freq = w_sweep_hat / (2*np.pi*T0)
        amps = resp_hat * Q0
        xlabel = "$f_d$ [Hz]"

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.viridis(np.linspace(0,1,len(f_amps_phys)))
    for j, F in enumerate(f_amps_phys):
        ax.plot(freq, np.zeros_like(freq)+j, amps[j,:,0],
                color=cmap[j], label=f"{F:.1f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("force idx"); ax.set_zlabel("|q|")
    ax.set_title("Force sweep – mode 1"); ax.legend(title="F")
    plt.tight_layout(); plt.show()
# ------------------------------------------------------------------------
