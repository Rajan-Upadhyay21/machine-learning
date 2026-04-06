import numpy as np
from scipy.stats import zscore

data = np.array([50, 52, 49, 51, 53, 90, 48, 50, 52])

z_scores = zscore(data)

print("Data:", data)
print("Z-Scores:", z_scores)
