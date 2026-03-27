# ---------------------------------------------------------
# Program: Train-Test Split Example
# Description:
# This program demonstrates how to split a dataset into
# training data and testing data using scikit-learn.
# This is one of the first and most important steps in
# a machine learning workflow.
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the sample Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Print dataset information
print("Total number of samples:", len(X))
print("Number of training samples:", len(X_train))
print("Number of testing samples:", len(X_test))
print("Training feature shape:", X_train.shape)
print("Testing feature shape:", X_test.shape)
