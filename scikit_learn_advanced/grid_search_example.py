# ---------------------------------------------------------
# Program: Grid Search Example
# Description:
# This program demonstrates hyperparameter tuning using
# GridSearchCV in scikit-learn.
# Grid search helps find the best parameter combination
# for a model.
# ---------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define parameter grid
param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

# Create model
model = SVC()

# Create grid search object
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

# Train grid search
grid_search.fit(X, y)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
