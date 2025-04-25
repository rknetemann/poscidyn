import os
import matplotlib.pyplot as plt
from pycobi import ODESystem

auto_dir    = "/home/raymo/Projects/auto-07p"
working_dir = os.path.join(os.path.dirname(__file__), "auto-07p")
os.makedirs(working_dir, exist_ok=True)

# ───────────────────────────────────────────────────────────────
#  compile the model (skip PyCobi's automatic first run)
# ───────────────────────────────────────────────────────────────
ode = ODESystem("oscillators",
                auto_dir=auto_dir,
                working_dir=working_dir,
                init_cont=False)


# ───────────────────────────────────────────────────────────────
#  1-parameter continuation of equilibria in  f1
# ───────────────────────────────────────────────────────────────
par_f1 = 33
results, _ = ode.run(
        IPS = 1,            # equilibria
        ICP = [par_f1],     # free parameter
        RL0 = 0.0, RL1 = 2.0,
        DS  = 1e-2, NMX = 400,
        ISP = 2, ISW = 1,   # detect HB, LP, …
)

# ───────────────────────────────────────────────────────────────
#  plot that branch
# ───────────────────────────────────────────────────────────────
ode.plot_continuation("PAR(33)", "U(1)", cont=0)
plt.xlabel("forcing amplitude  f1")
plt.ylabel("equilibrium  q1")
plt.show()
