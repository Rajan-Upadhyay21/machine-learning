from scipy.stats import ttest_1samp
import numpy as np

sample_data = np.array([52, 48, 50, 51, 49, 47, 53, 50, 52, 48])
population_mean = 50

t_stat, p_value = ttest_1samp(sample_data, population_mean)

print("Sample Data:", sample_data)
print("T-Statistic:", t_stat)
print("P-Value:", p_value)
