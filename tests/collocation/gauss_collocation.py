# Gauss–Legendre nodes/weights and collocation basis visualization
import numpy as np
import matplotlib.pyplot as plt

def gauss_legendre_nodes_weights(s, a=-1.0, b=1.0):
    """
    Return Gauss–Legendre nodes and weights on [a,b].
    """
    x, w = np.polynomial.legendre.leggauss(s)  # on [-1,1]
    # affine map [-1,1] -> [a,b]
    c = 0.5 * (x + 1) * (b - a) + a
    w_mapped = 0.5 * (b - a) * w
    return c, w_mapped

def lagrange_basis(x_nodes, x_eval):
    """
    Evaluate Lagrange basis {L_i} at points x_eval for given nodes x_nodes (1D).
    Returns matrix L of shape (len(x_nodes), len(x_eval)) where
    L[i, j] = L_i(x_eval[j]).
    """
    x_nodes = np.asarray(x_nodes)
    x_eval = np.asarray(x_eval)
    s = len(x_nodes)
    L = np.ones((s, x_eval.size))
    for i in range(s):
        for k in range(s):
            if k == i:
                continue
            L[i, :] *= (x_eval - x_nodes[k]) / (x_nodes[i] - x_nodes[k])
    return L

# --- Part 1: Nodes & weights (s = 5) on [0,1] ---
s1 = 15
nodes_01, weights_01 = gauss_legendre_nodes_weights(s1, a=0.0, b=1.0)

print("Gauss–Legendre nodes on [0,1] (s=5):\n", nodes_01)
print("Weights on [0,1] (s=5):\n", weights_01)

# Plot nodes and weights
plt.figure(figsize=(7, 4))
plt.scatter(nodes_01, np.zeros_like(nodes_01), s=80, label="Nodes")
for xi, wi in zip(nodes_01, weights_01):
    plt.vlines(xi, 0, wi, linewidth=2)
    plt.text(xi, wi + 0.005, f"{wi:.3f}", ha='centeIf the system is forced with known period, you usually omit TT from the unknowns (or keep it but constrain T=TfT=Tf​) and still need a phase condition to lock the time origin.r', fontsize=9)
plt.xlabel("x in [0,1]")
plt.ylabel("Weight")
plt.title("Gauss–Legendre Nodes and Weights (s = 5) on [0,1]")
plt.legend()
plt.tight_layout()
plt.show()

# --- Part 2: Lagrange collocation basis on [0,1] for s = 3 ---
s2 = 15
c3, _ = gauss_legendre_nodes_weights(s2, a=0.0, b=1.0)

# Evaluation grid
xg = np.linspace(0.0, 1.0, 400)
L = lagrange_basis(c3, xg)

# Plot basis functions L1, L2, L3
plt.figure(figsize=(7, 4))
for i in range(s2):
    plt.plot(xg, L[i, :], label=fr"$L_{i+1}(x)$")
# show Kronecker-delta property at nodes
plt.scatter(c3, np.ones_like(c3), s=60, marker='o', label="L_i(c_i)=1")
plt.scatter(np.repeat(c3, s2), np.zeros(s2*s2), s=15, marker='x', label="L_i(c_j)=0 (i≠j)")
plt.ylim(-0.2, 1.2)
plt.xlabel("x in [0,1]")
plt.ylabel("Basis value")
plt.title("Lagrange Collocation Basis at Gauss–Legendre Nodes (s = 3)")
plt.legend()
plt.tight_layout()
plt.show()

# --- Part 3: Example collocation polynomial (interpolant) ---
# Define a smooth function to approximate on the element
f = lambda x: np.cos(2*np.pi*x) + 0.2*np.sin(6*np.pi*x)
fg = f(xg)

# Interpolation values at nodes, then polynomial p(x) = sum f(c_i) L_i(x)
fc = f(c3)
p = (fc[:, None] * L).sum(axis=0)

# Plot function vs. collocation polynomial
plt.figure(figsize=(7, 4))
plt.plot(xg, fg, label="Target function f(x)")
plt.plot(xg, p, linestyle='--', label="Collocation polynomial p(x)")
plt.scatter(c3, fc, s=40, label="Node samples f(c_i)")
plt.xlabel("x in [0,1]")
plt.ylabel("Value")
plt.title("Function vs. Degree-2 Collocation Polynomial (s = 3)")
plt.legend()
plt.tight_layout()
plt.show()

# Notes (printed) to connect to periodic BVP collocation
print("\nNotes:")
print("- In Gauss collocation for ODEs/BVPs, the unknown solution on an element is expanded in this Lagrange basis.")
print("- The ODE residual is enforced at the Gauss nodes c_i, which yields the collocation equations for the stage values.")
print("- For periodic BVPs, elements tile the period domain; continuity is enforced at element boundaries, plus a phase condition and free period if needed.")
