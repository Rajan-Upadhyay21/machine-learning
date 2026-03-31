from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X, y = load_iris(return_X_y=True)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 3, 5, 10],
    "min_samples_split": [2, 4, 6]
}

model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X, y)

print("Best Parameters for Random Forest:")
print(grid_search.best_params_)

print("\nBest Score:")
print(grid_search.best_score_)
