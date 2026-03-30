import pandas as pd

# Sample numerical data
data = {
    "Height": [150, 160, 170, 180],
    "Weight": [50, 60, 70, 80]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Create interaction feature
df["Height_Weight_Interaction"] = df["Height"] * df["Weight"]

print("\nDataset After Creating Interaction Feature:")
print(df)
