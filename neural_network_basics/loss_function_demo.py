import numpy as np

# Example actual and predicted values
y_true_class = np.array([1, 0, 1, 1, 0])
y_pred_class = np.array([0.9, 0.2, 0.8, 0.6, 0.1])

y_true_reg = np.array([3.0, 5.0, 2.5, 7.0])
y_pred_reg = np.array([2.8, 4.9, 2.7, 6.5])

# Binary cross-entropy loss
bce_loss = -np.mean(
    y_true_class * np.log(y_pred_class) +
    (1 - y_true_class) * np.log(1 - y_pred_class)
)

# Mean squared error loss
mse_loss = np.mean((y_true_reg - y_pred_reg) ** 2)

print("Binary Cross-Entropy Loss:")
print(bce_loss)

print("\nMean Squared Error Loss:")
print(mse_loss)
