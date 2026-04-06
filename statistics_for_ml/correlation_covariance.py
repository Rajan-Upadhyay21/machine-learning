import numpy as np

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

covariance_matrix = np.cov(x, y)
correlation_matrix = np.corrcoef(x, y)

print("X:", x)
print("Y:", y)

print("\nCovariance Matrix:")
print(covariance_matrix)

print("\nCorrelation Matrix:")
print(correlation_matrix)
