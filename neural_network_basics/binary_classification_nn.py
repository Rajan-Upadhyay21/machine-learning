from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create binary classification dataset
X, y = make_classification(
    n_samples=300,
    n_features=6,
    n_classes=2,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create neural network classifier
model = MLPClassifier(
    hidden_layer_sizes=(12, 6),
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Binary Classification Accuracy:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
