from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

model = SVC()

grid_accuracy = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
grid_accuracy.fit(X, y)

grid_f1 = GridSearchCV(model, param_grid, cv=5, scoring="f1")
grid_f1.fit(X, y)

print("Best Parameters using Accuracy:")
print(grid_accuracy.best_params_)
print("Best Accuracy Score:")
print(grid_accuracy.best_score_)

print("\nBest Parameters using F1:")
print(grid_f1.best_params_)
print("Best F1 Score:")
print(grid_f1.best_score_)
