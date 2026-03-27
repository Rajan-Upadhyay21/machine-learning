# ---------------------------------------------------------
# Program: Numerical Integration Example
# Description:
# This program demonstrates integration using scipy.integrate.
# ---------------------------------------------------------

from scipy import integrate

# Function to integrate
def function(x):
    return x**2

# Integrate function from 0 to 3
result, error = integrate.quad(function, 0, 3)

print("Integration result:", result)
print("Estimated error:", error)
