# ---------------------------------------------------------
# Program: Sparse Matrix Example
# Description:
# This program demonstrates sparse matrix creation
# using scipy.sparse.
# ---------------------------------------------------------

import numpy as np
from scipy.sparse import csr_matrix

# Dense matrix
dense_matrix = np.array([
    [0, 0, 3],
    [4, 0, 0],
    [0, 0, 5]
])

# Convert to sparse matrix
sparse_matrix = csr_matrix(dense_matrix)

print("Dense matrix:")
print(dense_matrix)

print("\nSparse matrix representation:")
print(sparse_matrix)

print("\nSparse matrix converted back to dense:")
print(sparse_matrix.toarray())
