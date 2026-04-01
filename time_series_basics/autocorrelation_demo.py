import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot

dates = pd.date_range(start="2024-01-01", periods=20, freq="D")
values = [10, 12, 11, 13, 14, 15, 14, 16, 17, 18, 17, 19, 20, 21, 20, 22, 23, 24, 23, 25]

ts = pd.Series(values, index=dates)

autocorrelation_plot(ts)
plt.title("Autocorrelation Plot")
plt.show()
