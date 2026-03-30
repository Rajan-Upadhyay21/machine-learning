import pandas as pd

data = {
    "Marks": [45, 50, 52, 49, 300, 48, 51, 47, 46, 500]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

Q1 = df["Marks"].quantile(0.25)
Q3 = df["Marks"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nLower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

filtered_df = df[(df["Marks"] >= lower_bound) & (df["Marks"] <= upper_bound)]

print("\nDataset After Removing Outliers:")
print(filtered_df)
