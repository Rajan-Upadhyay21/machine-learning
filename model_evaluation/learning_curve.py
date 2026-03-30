import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Model
model = LogisticRegression(max_iter=200)

# Generate learning curve data
train_sizes, train_scores, test_scores = learning_curve(
    model, X, y, cv=5, train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0]
)

# Mean scores
train_scores_mean = train_scores.mean(axis=1)
test_scores_mean = test_scores.mean(axis=1)

# Plot
plt.plot(train_sizes, train_scores_mean, marker="o", label="Training Score")
plt.plot(train_sizes, test_scores_mean, marker="o", label="Validation Score")

plt.title("Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.show()
