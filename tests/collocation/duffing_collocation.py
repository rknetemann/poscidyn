# Re-run the final section to save the plot to a file instead of showing it interactively,
# to avoid any UI/display slowdowns.

import numpy as np
import matplotlib.pyplot as plt


w0     = 1.0
Q      = 20.0
gamma  = 1.0
fdrive = 1.0
omega  = 1.63
m      = 128
s      = 3
newton_maxit = 100
newton_tol   = 1e-8
fd_eps       = 1e-7

def gauss_nodes_s2():
    c1 = 0.5 - np.sqrt(3)/6.0
    c2 = 0.5 + np.sqrt(3)/6.0
    return np.array([c1, c2])

def lagrange_basis_and_deriv(nodes, x_eval):
    nodes = np.asarray(nodes)
    x_eval = np.asarray(x_eval)
    p = len(nodes)
    q = len(x_eval)
    L = np.ones((q, p))
    dLdx = np.zeros((q, p))
    for j in range(p):
        for k in range(p):
            if k == j: continue
            L[:, j] *= (x_eval - nodes[k])/(nodes[j]-nodes[k])
    for j in range(p):
        for k in range(p):
            if k == j: continue
            term = np.ones(q)/(nodes[j]-nodes[k])
            for l in range(p):
                if l==j or l==k: continue
                term *= (x_eval - nodes[l])/(nodes[j]-nodes[l])
            dLdx[:, j] += term
    return L, dLdx

T = 2.0*np.pi/omega
tau_nodes = np.linspace(0.0, 1.0, m+1)
h = np.diff(tau_nodes)
c = gauss_nodes_s2()
xi_nodes = np.concatenate(([0.0], c))
xi_colloc = c
xi_right  = np.array([1.0])
L_colloc, dL_colloc = lagrange_basis_and_deriv(xi_nodes, xi_colloc)
L_right,  dL_right  = lagrange_basis_and_deriv(xi_nodes, xi_right)

def F_t_u(tt, u):
    x, v = u
    dxdt = v
    dvdt = -(w0/Q)*v - (w0**2)*x - gamma*(x**3) + fdrive*np.cos(omega*tt)
    return np.array([dxdt, dvdt])

def G_tau_u(tau, u):
    return T * F_t_u(T * tau, u)

ndof_per_node = 2
nnodes_per_elem = 3
ndof_per_elem = ndof_per_node * nnodes_per_elem
ndof_total = m * ndof_per_elem

def pack_U(Ulist):
    return np.concatenate([Uk.reshape(-1) for Uk in Ulist])

def unpack_U(z):
    Ulist = []
    offset = 0
    for k in range(m):
        Uk = z[offset:offset+ndof_per_elem].reshape(nnodes_per_elem, ndof_per_node)
        Ulist.append(Uk)
        offset += ndof_per_elem
    return Ulist

def linear_guess(t):
    den = (w0**2 - omega**2)**2 + (w0*omega/Q)**2
    A = fdrive/np.sqrt(den)
    phi = np.arctan2((w0*omega/Q), (w0**2 - omega**2))
    x = A*np.cos(omega*t - phi)
    v = -A*omega*np.sin(omega*t - phi)
    return np.array([x, v])

U0_list = []
for k in range(m):
    tau_L = tau_nodes[k]
    tau_loc = tau_L + h[k] * xi_nodes
    t_loc = T * tau_loc
    Uk = np.stack([linear_guess(ti) for ti in t_loc], axis=0)
    U0_list.append(Uk)

z = pack_U(U0_list)

def element_right(Uk):
    return (L_right @ Uk)[0]

def element_collocation_values(Uk):
    return (L_colloc @ Uk)

def element_collocation_derivs(Uk, hk):
    dU_dxi = (dL_colloc @ Uk)
    return (1.0/hk) * dU_dxi

def residual(z):
    Ulist = unpack_U(z)
    R = []
    for k in range(m):
        Uk = Ulist[k]
        hk = h[k]
        tau_L = tau_nodes[k]
        Uc = element_collocation_values(Uk)
        dUc_dtau = element_collocation_derivs(Uk, hk)
        taus = tau_L + hk * c
        for i in range(2):
            Ri = dUc_dtau[i] - G_tau_u(taus[i], Uc[i])
            R.append(Ri)
    for k in range(m-1):
        Uk = Ulist[k]
        Uk1 = Ulist[k+1]
        right_k = element_right(Uk)
        cont = right_k - Uk1[0]
        R.append(cont)
    right_last = element_right(Ulist[-1])
    periodic = right_last - Ulist[0][0]
    R.append(periodic)
    return np.concatenate(R)

def jacobian_fd(z):
    R0 = residual(z)
    J = np.zeros((R0.size, z.size))
    for j in range(z.size):
        zj = z.copy()
        zj[j] += fd_eps
        R1 = residual(zj)
        J[:, j] = (R1 - R0)/fd_eps
    return J, R0

for it in range(1, newton_maxit+1):
    J, R0 = jacobian_fd(z)
    nrm = np.linalg.norm(R0, ord=np.inf)
    if nrm < newton_tol:
        break
    try:
        dz = np.linalg.solve(J, -R0)
    except np.linalg.LinAlgError:
        dz, *_ = np.linalg.lstsq(J, -R0, rcond=None)
    z = z + dz

Ulist = unpack_U(z)

# Prepare plot data
def lagrange_basis(nodes, x_eval):
    nodes = np.asarray(nodes)
    x_eval = np.asarray(x_eval)
    p = len(nodes); q = len(x_eval)
    L = np.ones((q, p))
    for j in range(p):
        for k in range(p):
            if k == j: continue
            L[:, j] *= (x_eval - nodes[k])/(nodes[j]-nodes[k])
    return L

c = np.array([0.5 - np.sqrt(3)/6.0, 0.5 + np.sqrt(3)/6.0])
xi_nodes = np.array([0.0, c[0], c[1]])

nplot_per_elem = 40
xi_plot = np.linspace(0.0, 1.0, nplot_per_elem)
L_plot = lagrange_basis(xi_nodes, xi_plot)

taus_plot = []
x_plot = []
for k in range(len(Ulist)):
    Uk = Ulist[k]
    tau_L = tau_nodes[k]
    tau_seg = tau_L + h[k] * xi_plot
    vals = (L_plot @ Uk)
    taus_plot.append(tau_seg)
    x_plot.append(vals[:, 0])

taus_plot = np.concatenate(taus_plot)
t_plot = T * taus_plot
x_plot = np.concatenate(x_plot)

order = np.argsort(t_plot)
t_plot = t_plot[order]
x_plot = x_plot[order]

# Save figure
plt.figure(figsize=(8,4))
plt.plot(t_plot, x_plot)
plt.xlabel("t")
plt.ylabel("x(t)")
plt.title("Duffing periodic orbit via Gauss collocation (s=2)")
plt.tight_layout()
plt.show()

# Print endpoint check
def element_right(Uk):
    L_right = lagrange_basis(xi_nodes, np.array([1.0]))
    return (L_right @ Uk)[0]

x0, v0 = Ulist[0][0]
xT, vT = element_right(Ulist[-1])
print(f"Endpoint check: x(0)={x0:.6f}, x(T)={xT:.6f}")
print(f"Period T = {T:.6f} (omega = {omega})")

# --- Amplitude stats over one period ---
imax = np.argmax(x_plot)
imin = np.argmin(x_plot)

x_max = x_plot[imax]
x_min = x_plot[imin]
t_at_xmax = t_plot[imax]
t_at_xmin = t_plot[imin]

amp_peak = np.max(np.abs(x_plot))     # max |x| over the period
amp_half_pp = 0.5 * (x_max - x_min)   # half peak-to-peak amplitude

print(f"Max x over period: {x_max:.6f} at t={t_at_xmax:.6f}")
print(f"Min x over period: {x_min:.6f} at t={t_at_xmin:.6f}")
print(f"Peak amplitude (max |x|): {amp_peak:.6f}")
print(f"Half peak-to-peak amplitude: {amp_half_pp:.6f}")