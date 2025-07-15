import numpy as np
X = np.random.rand(10, 5) # Create random data matrix

print(f"X:\n{X}")

U, S, VT = np.linalg.svd(X,full_matrices=False) 

print(f"Uhat:\n{U} \n\nShat:\n{S} \n\nVThat:\n{VT}")

# Method 1: Correct way to compute rank-r approximation
r = 2
Xapprox = U[:,:r] @ np.diag(S[:r]) @ VT[:r,:]

print(f"\nX_approx:\n{Xapprox}")

# Verify the reconstruction error decreases as r increases
print(f"Error with r={r}: {np.linalg.norm(X - Xapprox)}")

