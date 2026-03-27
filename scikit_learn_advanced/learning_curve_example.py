# ---------------------------------------------------------
# Program: Learning Curve Example
# Description:
# This program demonstrates how to measure model performance
# as the training set size increases.
# Learning curves help understand whether a model may benefit
# from more data or whether it is underfitting or overfitting.
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
import numpy as np

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create model
model = SVC(kernel="linear")

# Compute learning curve
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=model,
    X=X,
    y=y,
    cv=5,
    train_sizes=np.linspace(0.2, 1.0, 5)
)

# Print summarized results
print("Train sizes:")
print(train_sizes)

print("\nAverage training scores:")
print(train_scores.mean(axis=1))

print("\nAverage validation scores:")
print(validation_scores.mean(axis=1))
