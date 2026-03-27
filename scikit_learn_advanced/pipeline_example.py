# ---------------------------------------------------------
# Program: Pipeline Example
# Description:
# This program demonstrates a complete machine learning
# pipeline using StandardScaler and Support Vector Machine.
# Pipelines are useful because they combine preprocessing
# and modeling into one clean workflow.
# ---------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf"))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

print("Predicted labels:", predictions)
print("Actual labels:", y_test)
print("Pipeline accuracy:", accuracy_score(y_test, predictions))
