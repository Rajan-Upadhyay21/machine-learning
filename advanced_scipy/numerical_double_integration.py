# ---------------------------------------------------------
# Program: Numerical Double Integration
# Description:
# This program demonstrates double integration using
# scipy.integrate.dblquad.
# ---------------------------------------------------------

from scipy.integrate import dblquad

# Function to integrate
def function(y, x):
    return x * y

# Integrate over x in [0, 2], y in [0, 1]
result, error = dblquad(function, 0, 2, lambda x: 0, lambda x: 1)

print("Double integration result:", result)
print("Estimated error:", error)
