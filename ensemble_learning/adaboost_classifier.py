from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=42)

# Train model
adaboost_model.fit(X_train, y_train)

# Predict
y_pred = adaboost_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("AdaBoost Classifier Accuracy:", accuracy)
print("Predictions:", y_pred)
