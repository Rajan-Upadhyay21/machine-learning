# ---------------------------------------------------------
# Program: Root Finding Advanced
# Description:
# This program demonstrates solving an equation using
# scipy.optimize.root.
# ---------------------------------------------------------

from scipy.optimize import root

# Equation: x^3 - 2x - 5 = 0
def equation(x):
    return x**3 - 2*x - 5

# Initial guess
initial_guess = 2

# Solve for root
solution = root(equation, initial_guess)

print("Root finding result:")
print("Root value:", solution.x)
print("Success status:", solution.success)
print("Message:", solution.message)
