from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X, y = load_iris(return_X_y=True)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5, 10]
}

model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)

print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Estimator:")
print(grid_search.best_estimator_)

print("\nBest Score:")
print(grid_search.best_score_)
