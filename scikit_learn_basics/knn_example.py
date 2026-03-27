# ---------------------------------------------------------
# Program: K-Nearest Neighbors Example
# Description:
# This program demonstrates classification using
# KNeighborsClassifier from scikit-learn.
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print accuracy
print("Predicted labels:", predictions)
print("Actual labels:", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))
