# ---------------------------------------------------------
# Program: ODE Solver Example
# Description:
# This program solves an ordinary differential equation
# using scipy.integrate.solve_ivp.
# ---------------------------------------------------------

import numpy as np
from scipy.integrate import solve_ivp

# Differential equation: dy/dt = -2y
def differential_equation(t, y):
    return -2 * y

# Time span
t_span = (0, 5)

# Initial condition
y0 = [5]

# Evaluation points
t_eval = np.linspace(0, 5, 11)

# Solve ODE
solution = solve_ivp(differential_equation, t_span, y0, t_eval=t_eval)

print("Time points:")
print(solution.t)

print("\nSolution values:")
print(solution.y[0])
