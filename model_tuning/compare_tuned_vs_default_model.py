from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

default_model = RandomForestClassifier(random_state=42)
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_pred)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 3, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_pred)

print("Default Model Accuracy:")
print(default_accuracy)

print("\nTuned Model Accuracy:")
print(tuned_accuracy)

print("\nBest Parameters:")
print(grid_search.best_params_)
