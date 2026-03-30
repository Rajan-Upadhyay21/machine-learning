from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Create binary classification dataset
X, y = make_classification(
    n_samples=200, n_features=5, n_classes=2, random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_prob)

print("True Labels:")
print(y_test)

print("\nPredicted Probabilities:")
print(y_prob)

print("\nROC-AUC Score:")
print(roc_auc)
