from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Create model
model = LogisticRegression(max_iter=200)

# Perform 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross-Validation Scores:")
print(scores)

print("\nMean Cross-Validation Score:")
print(scores.mean())
