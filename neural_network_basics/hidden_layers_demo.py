from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model with one hidden layer
model_one_hidden = MLPClassifier(
    hidden_layer_sizes=(10,),
    max_iter=1000,
    random_state=42
)
model_one_hidden.fit(X_train, y_train)
y_pred_one = model_one_hidden.predict(X_test)
accuracy_one = accuracy_score(y_test, y_pred_one)

# Model with two hidden layers
model_two_hidden = MLPClassifier(
    hidden_layer_sizes=(20, 10),
    max_iter=1000,
    random_state=42
)
model_two_hidden.fit(X_train, y_train)
y_pred_two = model_two_hidden.predict(X_test)
accuracy_two = accuracy_score(y_test, y_pred_two)

print("Accuracy with One Hidden Layer:")
print(accuracy_one)

print("\nAccuracy with Two Hidden Layers:")
print(accuracy_two)
