from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train model
gb_regressor.fit(X_train, y_train)

# Predict
y_pred = gb_regressor.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)

print("Gradient Boosting Regressor Mean Squared Error:", mse)
print("First 10 Predictions:", y_pred[:10])
