# ---------------------------------------------------------
# Program: Statistical Testing Example
# Description:
# This program demonstrates a t-test using scipy.stats.
# ---------------------------------------------------------

from scipy import stats

# Sample datasets
sample1 = [23, 25, 28, 30, 32, 35, 36]
sample2 = [20, 22, 24, 26, 29, 31, 33]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)

print("T-statistic:", t_statistic)
print("P-value:", p_value)

if p_value < 0.05:
    print("The difference between samples is statistically significant.")
else:
    print("The difference between samples is not statistically significant.")
