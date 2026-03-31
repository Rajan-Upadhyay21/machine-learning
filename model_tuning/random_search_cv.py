from sklearn.datasets import load_iris
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

X, y = load_iris(return_X_y=True)

param_dist = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"]
}

model = SVC()

random_search = RandomizedSearchCV(
    model,
    param_distributions=param_dist,
    n_iter=5,
    cv=5,
    scoring="accuracy",
    random_state=42
)
random_search.fit(X, y)

print("Best Parameters:")
print(random_search.best_params_)

print("\nBest Cross-Validation Score:")
print(random_search.best_score_)
