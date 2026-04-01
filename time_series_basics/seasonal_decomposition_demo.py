import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

dates = pd.date_range(start="2024-01-01", periods=24, freq="M")
trend = np.arange(24) * 2
seasonal = [10, 12, 15, 14, 13, 12, 11, 10, 9, 11, 13, 14] * 2
noise = np.random.normal(0, 1, 24)

series = pd.Series(trend + seasonal + noise, index=dates)

result = seasonal_decompose(series, model="additive", period=12)
result.plot()
plt.show()
