from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
hierarchical_labels = AgglomerativeClustering(n_clusters=3).fit_predict(X)
dbscan_labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)

print("K-Means Silhouette Score:")
print(silhouette_score(X, kmeans_labels))

print("\nHierarchical Clustering Silhouette Score:")
print(silhouette_score(X, hierarchical_labels))

valid_mask = dbscan_labels != -1
if len(set(dbscan_labels[valid_mask])) > 1:
    print("\nDBSCAN Silhouette Score:")
    print(silhouette_score(X[valid_mask], dbscan_labels[valid_mask]))
else:
    print("\nDBSCAN Silhouette Score cannot be calculated properly for this clustering result.")
