from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import pandas as pd

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

model = DBSCAN(eps=0.8, min_samples=5)
labels = model.fit_predict(X)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Cluster"] = labels

print(df.head())
print("\nUnique Cluster Labels:")
print(df["Cluster"].unique())
