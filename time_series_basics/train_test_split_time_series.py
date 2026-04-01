import pandas as pd

dates = pd.date_range(start="2024-01-01", periods=12, freq="M")
values = [100, 110, 108, 115, 120, 125, 130, 128, 135, 140, 145, 150]

df = pd.DataFrame({"Value": values}, index=dates)

train = df.iloc[:9]
test = df.iloc[9:]

print("Training Data:")
print(train)

print("\nTesting Data:")
print(test)
