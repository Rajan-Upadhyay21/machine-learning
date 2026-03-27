# ---------------------------------------------------------
# Program: Basic Linear Algebra with SciPy
# Description:
# This program demonstrates matrix operations using
# scipy.linalg.
# ---------------------------------------------------------

import numpy as np
from scipy import linalg

# Create matrix
A = np.array([[1, 2], [3, 4]])

print("Matrix A:")
print(A)

# Determinant
print("\nDeterminant of A:", linalg.det(A))

# Inverse
print("\nInverse of A:")
print(linalg.inv(A))

# Eigenvalues
print("\nEigenvalues of A:")
print(linalg.eigvals(A))
