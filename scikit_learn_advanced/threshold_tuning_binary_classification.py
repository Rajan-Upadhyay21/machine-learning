# ---------------------------------------------------------
# Program: Threshold Tuning for Binary Classification
# Description:
# This program demonstrates how classification decisions
# can change when the probability threshold is adjusted.
# This is useful when precision and recall trade-offs matter.
# ---------------------------------------------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

# Create binary dataset
X, y = make_classification(
    n_samples=500,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Get predicted probabilities for positive class
probabilities = model.predict_proba(X_test)[:, 1]

# Apply custom threshold
threshold = 0.3
custom_predictions = (probabilities >= threshold).astype(int)

print("Threshold used:", threshold)
print("Precision:", precision_score(y_test, custom_predictions))
print("Recall:", recall_score(y_test, custom_predictions))
