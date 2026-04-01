import pandas as pd
import numpy as np

dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
values = np.random.randint(100, 200, size=30)

df = pd.DataFrame({"Sales": values}, index=dates)

weekly_sum = df.resample("W").sum()

print("Daily Data:")
print(df.head())

print("\nWeekly Resampled Data:")
print(weekly_sum)
