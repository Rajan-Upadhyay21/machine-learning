from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

model = SVC()

grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X, y)

print("Best Parameters for SVM:")
print(grid_search.best_params_)

print("\nBest Score:")
print(grid_search.best_score_)
