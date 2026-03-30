import numpy as np
import matplotlib.pyplot as plt

# Sample input values
x = np.linspace(-10, 10, 200)

# Activation functions
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)
relu = np.maximum(0, x)

# Plot sigmoid
plt.figure(figsize=(8, 5))
plt.plot(x, sigmoid)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Plot tanh
plt.figure(figsize=(8, 5))
plt.plot(x, tanh)
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

# Plot ReLU
plt.figure(figsize=(8, 5))
plt.plot(x, relu)
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()
