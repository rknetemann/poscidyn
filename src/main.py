# ───────────────────────── main.py ──────────────────────────
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401

from models import PhysicalModel, NonDimensionalisedModel
from nonlinear_dynamics import NonlinearDynamics

# ────────────── switches ────────────────────────────────────
RUN_TIME   = False     # single-tone time trace
RUN_FREQ   = True      # frequency-response curve
RUN_FORCE  = False     # force-sweep surface

# ────────────── build & scale model ─────────────────────────
N   = 1
mdl = PhysicalModel.from_example(N).non_dimensionalise()
nld = NonlinearDynamics(mdl)
#mdl = Model.from_random(N)

# ────────────── eigenfrequencies ─────────────────────────

eigenfreq = mdl.omega_0_hat
quality_factors = mdl.Q
x_ref = mdl.x_ref
print(f"Eigenfrequencies: {eigenfreq}")
print(f"Quality factors: {quality_factors}")
print(f"Non-dimensionalised x_ref: {x_ref}")

# =============== frequency sweep ===================
if RUN_FREQ:
    print("\nCalculating frequency response…")
    tau_end = 1000
    y0_hat = jnp.zeros(2 * N)
    y0_hat = y0_hat.at[0].set(.0)
    y0_hat = y0_hat.at[1].set(0)
    
    F_omega_hat, q_steady, q_steady_total, _ = nld.frequency_response(
        tau_end=tau_end, y0_hat=y0_hat
    )
    
    plt.figure(figsize=(7,4))
    for m in range(N):
        plt.plot(F_omega_hat, q_steady[:, m], label=f"Mode {m+1}")
    for f in eigenfreq:
        plt.axvline(f, ls="--", color="r", alpha=.6)
    plt.plot(F_omega_hat, q_steady_total, label="Total Response", color="k", lw=2, alpha=0.8)
    plt.xlabel("Non-dimensionalized drive frequency"); plt.ylabel("Non-dimensionalized amplitude")
    plt.title("Frequency response"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.show()