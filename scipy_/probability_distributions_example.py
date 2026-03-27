# ---------------------------------------------------------
# Program: Probability Distributions Example
# Description:
# This program demonstrates normal distribution functions
# using scipy.stats.
# ---------------------------------------------------------

from scipy.stats import norm

# Mean and standard deviation
mean = 0
std_dev = 1

# Probability density function
pdf_value = norm.pdf(0, loc=mean, scale=std_dev)

# Cumulative distribution function
cdf_value = norm.cdf(1.96, loc=mean, scale=std_dev)

print("PDF at x = 0:", pdf_value)
print("CDF at x = 1.96:", cdf_value)
