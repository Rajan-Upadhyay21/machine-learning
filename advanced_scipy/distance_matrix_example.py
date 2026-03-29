# ---------------------------------------------------------
# Program: Distance Matrix Example
# Description:
# This program computes pairwise Euclidean distances
# using scipy.spatial.distance.cdist.
# ---------------------------------------------------------

import numpy as np
from scipy.spatial.distance import cdist

# Sample points
points_a = np.array([
    [0, 0],
    [1, 1],
    [2, 2]
])

points_b = np.array([
    [1, 0],
    [2, 1]
])

# Compute distance matrix
distance_matrix = cdist(points_a, points_b, metric="euclidean")

print("Points A:")
print(points_a)

print("\nPoints B:")
print(points_b)

print("\nPairwise distance matrix:")
print(distance_matrix)
