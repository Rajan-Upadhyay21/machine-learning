from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
import pandas as pd

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

model = GaussianMixture(n_components=3, random_state=42)
labels = model.fit_predict(X)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Cluster"] = labels

print(df.head())
print("\nCluster Means:")
print(model.means_)
