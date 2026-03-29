# ---------------------------------------------------------
# Program: Curve Fitting Example
# Description:
# This program fits a curve to data using scipy.optimize.curve_fit.
# ---------------------------------------------------------

import numpy as np
from scipy.optimize import curve_fit

# Define model function
def model_function(x, a, b):
    return a * x + b

# Sample data
x_data = np.array([1, 2, 3, 4, 5, 6])
y_data = np.array([2.1, 4.2, 6.1, 8.3, 10.2, 12.1])

# Fit curve
parameters, covariance = curve_fit(model_function, x_data, y_data)

print("Best-fit parameters:")
print("a =", parameters[0])
print("b =", parameters[1])

# Predicted values
predicted = model_function(x_data, *parameters)

print("\nPredicted values:")
print(predicted)
