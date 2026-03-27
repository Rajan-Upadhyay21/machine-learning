# ---------------------------------------------------------
# Program: Standard Scaler Example
# Description:
# This program demonstrates feature scaling using
# StandardScaler from scikit-learn.
# Feature scaling is useful because it brings features
# to a common scale, which helps many ML algorithms.
# ---------------------------------------------------------

from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.array([
    [1, 200],
    [2, 300],
    [3, 400],
    [4, 500]
])

# Create scaler object
scaler = StandardScaler()

# Fit and transform the data
scaled_X = scaler.fit_transform(X)

# Print original and scaled data
print("Original data:")
print(X)

print("\nScaled data:")
print(scaled_X)
