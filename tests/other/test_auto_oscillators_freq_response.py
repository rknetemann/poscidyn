import os, numpy as np, matplotlib.pyplot as plt
from pycobi import ODESystem

auto_dir    = "/home/raymo/Projects/auto-07p"
work_dir    = os.path.join(os.path.dirname(__file__), "auto-07p")
os.makedirs(work_dir, exist_ok=True)

ode = ODESystem("oscillators", auto_dir=auto_dir,
                working_dir=work_dir, init_cont=False)

# ─── parameter indices (N = 2) ──────────────────────────────────────
par_f1   = 33          # first forcing amplitude
par_om   = 35          # excitation frequency ω   (last parameter)

# ─── STEP 1 : build a tiny limit-cycle by increasing f1 ──────────────
lc0, _ = ode.run(
        IPS = 2, ICP = [par_f1],
        DS  = 1e-4, RL0 = 0.0, RL1 = 0.05,
        NMX = 200, ISW = 1, ILP = 1)

# grab the LAB number (column is a 2-level MultiIndex → ('LAB',''))
lab_seed = lc0[('LAB', '')].iat[-1]

# ─── STEP 2 : freeze f1, sweep the frequency ω ───────────────────────
lc_om, _ = ode.run(
        origin = lab_seed,       # ← works now
        IPS    = 2,
        ICP    = [par_om],
        DS     = 1e-2,
        RL0    = 0.2, RL1 = 3.0,
        NMX    = 400, ISW = 1, ILP = 1)

# plot amplitude curve
ode.plot_continuation("PAR(35)", "max(U(1))", cont=lc_om)
plt.xlabel("excitation frequency  ω")
plt.ylabel("steady-state amplitude  max q₁")
plt.show()