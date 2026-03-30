import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample numerical data
data = {
    "Age": [20, 25, 30, 35, 40],
    "Salary": [30000, 50000, 70000, 90000, 110000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Standard Scaling
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

standard_scaled_df = pd.DataFrame(standard_scaled, columns=df.columns)
print("\nStandard Scaled Data:")
print(standard_scaled_df)

# Min-Max Scaling
minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

minmax_scaled_df = pd.DataFrame(minmax_scaled, columns=df.columns)
print("\nMin-Max Scaled Data:")
print(minmax_scaled_df)
