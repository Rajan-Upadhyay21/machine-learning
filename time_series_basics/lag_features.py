import pandas as pd

dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
values = [50, 55, 53, 60, 62, 65, 67, 70, 72, 75]

df = pd.DataFrame({"Sales": values}, index=dates)

df["Lag_1"] = df["Sales"].shift(1)
df["Lag_2"] = df["Sales"].shift(2)

print(df)
