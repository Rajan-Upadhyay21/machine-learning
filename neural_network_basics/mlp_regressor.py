from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create MLP regressor
model = MLPRegressor(
    hidden_layer_sizes=(50, 25),
    max_iter=1000,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

print("First 10 Actual Values:")
print(y_test[:10])

print("\nFirst 10 Predicted Values:")
print(y_pred[:10])

print("\nMLP Regressor Mean Squared Error:")
print(mse)
