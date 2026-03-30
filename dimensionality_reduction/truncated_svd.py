from sklearn.decomposition import TruncatedSVD
import numpy as np
import pandas as pd

# Sample matrix data
X = np.array([
    [1, 0, 0, 1, 2],
    [0, 1, 1, 0, 1],
    [1, 1, 0, 1, 3],
    [0, 0, 1, 1, 1]
])

# Apply Truncated SVD
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

df_svd = pd.DataFrame(X_svd, columns=["Component1", "Component2"])

print("Original Shape:", X.shape)
print("Reduced Shape:", X_svd.shape)
print("\nTruncated SVD Output:")
print(df_svd)
