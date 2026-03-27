# ---------------------------------------------------------
# Program: Optimization Intro
# Description:
# This program demonstrates basic function minimization
# using scipy.optimize.
# ---------------------------------------------------------

from scipy.optimize import minimize

# Function to minimize
def objective_function(x):
    return x**2 + 4*x + 4

# Initial guess
initial_guess = [0]

# Minimize function
result = minimize(objective_function, initial_guess)

print("Optimization result:")
print("Minimum value:", result.fun)
print("Best x:", result.x)
