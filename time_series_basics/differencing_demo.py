import pandas as pd

dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
values = [100, 105, 110, 120, 130, 140, 150, 160, 170, 180]

ts = pd.Series(values, index=dates)

differenced = ts.diff()

print("Original Series:")
print(ts)

print("\nDifferenced Series:")
print(differenced)
