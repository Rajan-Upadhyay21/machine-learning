# ---------------------------------------------------------
# Program: Stacking Classifier Example
# Description:
# This program demonstrates ensemble learning using
# StackingClassifier in scikit-learn.
# Stacking combines predictions from multiple base models
# and trains a final model on top of them.
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define base estimators
estimators = [
    ("knn", KNeighborsClassifier(n_neighbors=3)),
    ("rf", RandomForestClassifier(random_state=42)),
    ("svc", SVC(probability=True, random_state=42))
]

# Define stacking classifier
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200)
)

# Train model
stacking_model.fit(X_train, y_train)

# Make predictions
predictions = stacking_model.predict(X_test)

# Print results
print("Predicted labels:", predictions)
print("Actual labels:", y_test)
print("Stacking classifier accuracy:", accuracy_score(y_test, predictions))
