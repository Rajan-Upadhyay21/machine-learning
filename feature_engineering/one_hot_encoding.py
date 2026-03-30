import pandas as pd

# Sample categorical data
data = {
    "Color": ["Red", "Blue", "Green", "Blue", "Red"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Apply One-Hot Encoding
encoded_df = pd.get_dummies(df, columns=["Color"])

print("\nDataset After One-Hot Encoding:")
print(encoded_df)
