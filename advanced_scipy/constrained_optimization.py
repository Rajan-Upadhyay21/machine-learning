# ---------------------------------------------------------
# Program: Constrained Optimization
# Description:
# This program demonstrates constrained minimization using
# scipy.optimize.minimize with bounds and constraints.
# ---------------------------------------------------------

from scipy.optimize import minimize

# Objective function
def objective_function(x):
    return x[0]**2 + x[1]**2

# Constraint: x + y >= 1
constraints = ({
    "type": "ineq",
    "fun": lambda x: x[0] + x[1] - 1
})

# Bounds for variables
bounds = [(0, None), (0, None)]

# Initial guess
initial_guess = [0.5, 0.5]

# Solve optimization problem
result = minimize(
    objective_function,
    initial_guess,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

print("Optimization success:", result.success)
print("Optimal solution:", result.x)
print("Minimum objective value:", result.fun)
