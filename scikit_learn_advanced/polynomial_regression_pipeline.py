# ---------------------------------------------------------
# Program: Polynomial Regression Pipeline
# Description:
# This program demonstrates polynomial regression using
# a scikit-learn pipeline.
# It expands features and then fits a linear regression model.
# ---------------------------------------------------------

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([1, 4, 9, 16, 25, 36])

# Create polynomial regression pipeline
pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2)),
    ("regressor", LinearRegression())
])

# Train model
pipeline.fit(X, y)

# Predict values
predictions = pipeline.predict(X)

print("Input values:")
print(X.flatten())

print("\nActual output values:")
print(y)

print("\nPredicted output values:")
print(predictions)
