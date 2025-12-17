import numpy as np

# Figure 1 from Nonlinear dynamic characterization of twodimensional materials by D. Davidovikj...

# It is exfoliated graphene based on Experimental characterization of graphene by electrostatic resonance frequency tuning. They seem to have the same resonator.

gamma_simulation = 1.00e-05
x_ref_simulation = 113.27
omega_ref_simulation = 1.0

R = 2.5e-06
h = 5.0e-09
rho = 2250.0
k3 = 1.35e15
m = np.pi * R**2 * h * rho
m = 5.7e-06 * np.pi * R**2 # Table 1 from Dynamics of 2D material membranes by P.G. Steeneken et al. 
print(m)
m_eff = 0.269 * m

x_ref_paper = 5.0e-09
omega_ref_paper = 14.7 * 1.0e6 * 2 * np.pi
gamma_paper = k3 / m_eff

print("Dimensionless gamma:")
gamma_dimensionless = gamma_paper * x_ref_paper**2 / omega_ref_paper**2
print(f"- Paper: {gamma_dimensionless:.2e}")

gamma_dimensionless_estimated = gamma_simulation * x_ref_simulation**2 / omega_ref_simulation**2
print(f"- Estimated: {gamma_dimensionless_estimated:.2e}")

print("\nPhysical gamma:")
gamma = k3 / m_eff
print(f"- Paper: {gamma:.2e}")

gamma_estimated = gamma_dimensionless_estimated * omega_ref_paper**2 / x_ref_paper**2
print(f"- Estimated: {gamma_estimated:.2e}")

estimation_error = abs(gamma - gamma_estimated) / gamma * 100.0
print(f"\nEstimation error: {estimation_error:.2f} %")