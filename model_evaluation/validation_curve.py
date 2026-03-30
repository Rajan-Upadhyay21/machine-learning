import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Parameter range
param_range = [0.01, 0.1, 1, 10, 100]

# Generate validation curve data
train_scores, test_scores = validation_curve(
    LogisticRegression(max_iter=200),
    X,
    y,
    param_name="C",
    param_range=param_range,
    cv=5
)

# Mean scores
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# Plot
plt.plot(param_range, train_scores_mean, marker="o", label="Training Score")
plt.plot(param_range, test_scores_mean, marker="o", label="Validation Score")

plt.title("Validation Curve")
plt.xlabel("C Parameter")
plt.ylabel("Score")
plt.xscale("log")
plt.legend()
plt.grid(True)
plt.show()
