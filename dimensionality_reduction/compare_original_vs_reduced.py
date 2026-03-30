from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
X = iris.data

print("Original Data Shape:", X.shape)

# Apply PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print("Reduced Data Shape:", X_reduced.shape)

print("\nFirst 5 rows of original data:")
print(X[:5])

print("\nFirst 5 rows of reduced data:")
print(X_reduced[:5])
