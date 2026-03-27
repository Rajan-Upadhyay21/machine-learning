# ---------------------------------------------------------
# Program: ROC AUC Multiclass Example
# Description:
# This program demonstrates probability-based evaluation
# using ROC AUC score for a multiclass classification problem.
# ---------------------------------------------------------

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict probabilities
predicted_probabilities = model.predict_proba(X_test)

# Compute ROC AUC score
auc_score = roc_auc_score(
    y_test,
    predicted_probabilities,
    multi_class="ovr"
)

print("Predicted probability shape:", predicted_probabilities.shape)
print("Multiclass ROC AUC score:", auc_score)
