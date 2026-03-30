from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd

# Load dataset
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Convert to DataFrame
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

print("Original Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)
print("\nPCA Output:")
print(df_pca.head())
