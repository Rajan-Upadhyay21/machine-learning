# ---------------------------------------------------------
# Program: PCA Example
# Description:
# This program demonstrates dimensionality reduction using
# Principal Component Analysis (PCA).
# PCA reduces the number of features while preserving
# as much information as possible.
# ---------------------------------------------------------

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Original feature shape:", X.shape)
print("Reduced feature shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("First five transformed rows:")
print(X_reduced[:5])
