import pandas as pd

dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
values = [100, 102, 101, 105, 107, 110, 108, 111, 115, 117]

ts = pd.Series(values, index=dates)

moving_avg = ts.rolling(window=3).mean()
forecast = moving_avg.iloc[-1]

print("Original Series:")
print(ts)

print("\n3-Day Moving Average:")
print(moving_avg)

print("\nNext Forecast Value Based on Moving Average:")
print(forecast)
