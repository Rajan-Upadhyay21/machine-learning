# ---------------------------------------------------------
# Program: Eigen Decomposition Advanced
# Description:
# This program computes eigenvalues and eigenvectors
# using scipy.linalg.eig.
# ---------------------------------------------------------

import numpy as np
from scipy.linalg import eig

# Matrix
A = np.array([
    [4, 2],
    [1, 3]
])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eig(A)

print("Matrix A:")
print(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)
