from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data

print("Dataset Shape:")
print(X.shape)

print("\nNumber of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

if X.shape[1] > 2:
    print("\nDimensionality reduction can be useful because the dataset has more than 2 features.")
else:
    print("\nDimensionality reduction may not be necessary.")
