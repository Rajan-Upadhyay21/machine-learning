# ---------------------------------------------------------
# Program: Sparse Linear Solver
# Description:
# This program solves a sparse linear system using
# scipy.sparse.linalg.spsolve.
# ---------------------------------------------------------

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Sparse coefficient matrix
A = csr_matrix([
    [3, 0, 1],
    [0, 4, 0],
    [1, 0, 2]
])

# Right-hand side
b = np.array([9, 8, 5])

# Solve sparse system
solution = spsolve(A, b)

print("Sparse matrix A:")
print(A)

print("\nRight-hand side vector b:")
print(b)

print("\nSolution:")
print(solution)
