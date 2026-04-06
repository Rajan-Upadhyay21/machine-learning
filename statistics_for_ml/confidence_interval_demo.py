import numpy as np
from scipy import stats

data = np.array([45, 50, 52, 48, 49, 51, 47, 53, 46, 50])

mean_value = np.mean(data)
std_error = stats.sem(data)
confidence_interval = stats.t.interval(
    confidence=0.95,
    df=len(data) - 1,
    loc=mean_value,
    scale=std_error
)

print("Data:", data)
print("Mean:", mean_value)
print("95% Confidence Interval:", confidence_interval)
