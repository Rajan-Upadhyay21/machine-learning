import numpy as np

# Input values
X = np.array([1.0, 2.0])

# Weights and biases
weights_hidden = np.array([[0.5, -0.6], [0.8, 0.2]])
bias_hidden = np.array([0.1, 0.2])

weights_output = np.array([[0.7], [-1.2]])
bias_output = np.array([0.3])

# ReLU activation
def relu(x):
    return np.maximum(0, x)

# Forward propagation
hidden_input = np.dot(X, weights_hidden) + bias_hidden
hidden_output = relu(hidden_input)

final_input = np.dot(hidden_output, weights_output) + bias_output
final_output = relu(final_input)

print("Input Layer:")
print(X)

print("\nHidden Layer Input:")
print(hidden_input)

print("\nHidden Layer Output:")
print(hidden_output)

print("\nOutput Layer Input:")
print(final_input)

print("\nFinal Output:")
print(final_output)
