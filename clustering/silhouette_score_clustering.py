from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

score = silhouette_score(X, labels)

print("Silhouette Score:")
print(score)
