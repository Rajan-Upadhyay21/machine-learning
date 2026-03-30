from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
log_model = LogisticRegression(max_iter=200)
tree_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Voting classifier
voting_model = VotingClassifier(
    estimators=[
        ("lr", log_model),
        ("dt", tree_model),
        ("rf", rf_model)
    ],
    voting="hard"
)

# Train model
voting_model.fit(X_train, y_train)

# Predict
y_pred = voting_model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)

print("Voting Classifier Accuracy:", accuracy)
print("Predictions:", y_pred)
