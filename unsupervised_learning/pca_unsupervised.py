from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

iris = load_iris()
X = iris.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

print(df.head())
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)
