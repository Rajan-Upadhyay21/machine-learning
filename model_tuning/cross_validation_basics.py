from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)

model = LogisticRegression(max_iter=200)

scores = cross_val_score(model, X, y, cv=5)

print("Cross-Validation Scores:")
print(scores)

print("\nMean Cross-Validation Score:")
print(scores.mean())
