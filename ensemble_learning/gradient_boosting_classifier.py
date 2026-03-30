from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train model
gb_classifier.fit(X_train, y_train)

# Predict
y_pred = gb_classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Gradient Boosting Classifier Accuracy:", accuracy)
print("Predictions:", y_pred)
