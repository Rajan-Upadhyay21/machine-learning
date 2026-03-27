# ---------------------------------------------------------
# Program: Randomized Search Example
# Description:
# This program demonstrates hyperparameter tuning using
# RandomizedSearchCV.
# Randomized search is useful when the parameter space
# is large and you want a faster search than grid search.
# ---------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define parameter distribution
param_dist = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3]
}

# Create model
model = RandomForestClassifier(random_state=42)

# Create randomized search object
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    random_state=42,
    scoring="accuracy"
)

# Train randomized search
random_search.fit(X, y)

print("Best parameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)
