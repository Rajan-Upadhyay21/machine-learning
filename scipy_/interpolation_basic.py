# ---------------------------------------------------------
# Program: Interpolation Basic
# Description:
# This program demonstrates interpolation using scipy.interpolate.
# ---------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d

# Original data points
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])

# Create interpolation function
interpolation_function = interp1d(x, y, kind="linear")

# New points
new_x = np.array([1.5, 2.5, 3.5])
new_y = interpolation_function(new_x)

print("New x values:", new_x)
print("Interpolated y values:", new_y)
