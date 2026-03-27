# ---------------------------------------------------------
# Program: Feature Importance with Random Forest
# Description:
# This program demonstrates how to extract and inspect
# feature importance values from a Random Forest model.
# Feature importance helps understand which input features
# influence the model the most.
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Extract feature importance
importances = model.feature_importances_

print("Feature importance values:")
for feature_name, importance in zip(feature_names, importances):
    print(f"{feature_name}: {importance:.4f}")
