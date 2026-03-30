import pandas as pd

# Sample continuous data
data = {
    "Age": [12, 18, 25, 35, 45, 60, 72]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Define bins and labels
bins = [0, 18, 35, 60, 100]
labels = ["Child", "Young Adult", "Adult", "Senior"]

df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

print("\nDataset After Binning:")
print(df)
