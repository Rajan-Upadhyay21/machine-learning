import pandas as pd
import numpy as np

dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
values = [10, 12, 14, 13, 15, 18, 20, 19, 21, 23]

ts = pd.Series(values, index=dates)

rolling_avg = ts.rolling(window=3).mean()

print("Original Series:")
print(ts)

print("\nRolling Mean:")
print(rolling_avg)
