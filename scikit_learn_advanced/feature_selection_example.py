# ---------------------------------------------------------
# Program: Feature Selection Example
# Description:
# This program demonstrates selecting the most important
# features using SelectKBest.
# Feature selection helps reduce irrelevant features and
# improve model simplicity and performance.
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Select top 2 features
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print("Original feature shape:", X.shape)
print("Selected feature shape:", X_selected.shape)
print("Feature scores:", selector.scores_)
print("Selected feature indices:", selector.get_support(indices=True))
