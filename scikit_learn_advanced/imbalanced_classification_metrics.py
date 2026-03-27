# ---------------------------------------------------------
# Program: Imbalanced Classification Metrics
# Description:
# This program demonstrates why accuracy alone is not enough
# for imbalanced classification problems.
# It shows precision, recall, and F1-score.
# ---------------------------------------------------------

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Example imbalanced case
y_true = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
y_pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
