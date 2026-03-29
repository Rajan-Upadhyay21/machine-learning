# ---------------------------------------------------------
# Program: Solve Linear System
# Description:
# This program solves a system of linear equations using
# scipy.linalg.solve.
# ---------------------------------------------------------

import numpy as np
from scipy.linalg import solve

# Coefficient matrix
A = np.array([
    [3, 2, -1],
    [2, -2, 4],
    [-1, 0.5, -1]
])

# Right-hand side vector
b = np.array([1, -2, 0])

# Solve system
solution = solve(A, b)

print("Coefficient matrix A:")
print(A)

print("\nRight-hand side vector b:")
print(b)

print("\nSolution of the system:")
print(solution)
