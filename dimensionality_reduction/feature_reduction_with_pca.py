from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data
feature_names = iris.feature_names

# Original DataFrame
df_original = pd.DataFrame(X, columns=feature_names)
print("Original Dataset:")
print(df_original.head())

# Reduce to 2 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

df_reduced = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
print("\nReduced Dataset with PCA:")
print(df_reduced.head())
