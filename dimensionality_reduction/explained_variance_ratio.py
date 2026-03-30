from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
X = iris.data

# Fit PCA
pca = PCA()
pca.fit(X)

print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nCumulative Explained Variance:")
cumulative = pca.explained_variance_ratio_.cumsum()
print(cumulative)
