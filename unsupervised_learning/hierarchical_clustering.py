from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

model = AgglomerativeClustering(n_clusters=3)
labels = model.fit_predict(X)

df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Cluster"] = labels

print(df.head())
