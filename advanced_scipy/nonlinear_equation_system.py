# ---------------------------------------------------------
# Program: Nonlinear Equation System
# Description:
# This program solves a system of nonlinear equations
# using scipy.optimize.fsolve.
# ---------------------------------------------------------

from scipy.optimize import fsolve

# Define system of equations
def equations(variables):
    x, y = variables
    eq1 = x**2 + y**2 - 4
    eq2 = x - y - 1
    return [eq1, eq2]

# Initial guess
initial_guess = [1, 1]

# Solve system
solution = fsolve(equations, initial_guess)

print("Solution of nonlinear system:")
print("x =", solution[0])
print("y =", solution[1])
