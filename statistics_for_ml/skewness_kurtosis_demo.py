import numpy as np
from scipy.stats import skew, kurtosis

data = np.array([10, 12, 13, 15, 18, 21, 24, 30, 45])

skewness_value = skew(data)
kurtosis_value = kurtosis(data)

print("Data:", data)
print("Skewness:", skewness_value)
print("Kurtosis:", kurtosis_value)
