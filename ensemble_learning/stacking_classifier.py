from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
estimators = [
    ("lr", LogisticRegression(max_iter=200)),
    ("dt", DecisionTreeClassifier(random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=100, random_state=42))
]

# Meta model
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200)
)

# Train model
stacking_model.fit(X_train, y_train)

# Predict
y_pred = stacking_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Stacking Classifier Accuracy:", accuracy)
print("Predictions:", y_pred)
