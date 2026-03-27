# ---------------------------------------------------------
# Program: Cross Validation Example
# Description:
# This program demonstrates how to evaluate a model using
# cross-validation in scikit-learn.
# Cross-validation is important because it gives a more
# reliable estimate of model performance than a single split.
# ---------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create model
model = LogisticRegression(max_iter=200)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())
print("Highest score:", scores.max())
print("Lowest score:", scores.min())
