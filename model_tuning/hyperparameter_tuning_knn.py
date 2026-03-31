from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)

param_grid = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}

model = KNeighborsClassifier()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X, y)

print("Best Parameters for KNN:")
print(grid_search.best_params_)

print("\nBest Score:")
print(grid_search.best_score_)
