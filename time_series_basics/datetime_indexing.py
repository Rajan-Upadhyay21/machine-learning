import pandas as pd
import numpy as np

dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
values = np.random.randint(10, 50, size=10)

df = pd.DataFrame({"Value": values}, index=dates)

print("Original Data:")
print(df)

print("\nData for First 5 Days:")
print(df["2024-01-01":"2024-01-05"])
