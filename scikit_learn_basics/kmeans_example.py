# ---------------------------------------------------------
# Program: K-Means Clustering Example
# Description:
# This program demonstrates unsupervised learning using
# KMeans clustering from scikit-learn.
# ---------------------------------------------------------

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

# Create and train the model
model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

# Print clustering results
print("Cluster labels:")
print(model.labels_)

print("\nCluster centers:")
print(model.cluster_centers_)
