import os
import sys
from pycobi import ODESystem
import matplotlib.pyplot as plt
import numpy as np

model = "oscillators.eq"

# installation directory of auto-07p
auto_dir = "/home/raymo/Projects/auto-07p"

# path to working directory for auto-07p
working_dir = os.path.join(os.path.dirname(__file__), "auto-07p")
os.makedirs(working_dir, exist_ok=True)

ode = ODESystem(model, auto_dir=auto_dir, working_dir=working_dir)

ode = ODESystem.from_yaml(
    "model_templates.neural_mass_models.qif.qif_sfa", auto_dir=auto_dir, working_dir=working_dir,
    node_vars={'p/qif_sfa_op/Delta': 2.0, 'p/qif_sfa_op/alpha': 1.0, 'p/qif_sfa_op/eta': 3.0},
    edge_vars=[('p/qif_sfa_op/r', 'p/qif_sfa_op/r_in', {'weight': 15.0*np.sqrt(2.0)})],
    NPR=100, NMX=30000
)

ode.plot_continuation("t", "p/qif_sfa_op/r", cont=0)
plt.show()