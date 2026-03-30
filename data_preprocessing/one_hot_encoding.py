import pandas as pd

data = {
    "Color": ["Red", "Blue", "Green", "Blue", "Red"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

encoded_df = pd.get_dummies(df, columns=["Color"])

print("\nDataset After One-Hot Encoding:")
print(encoded_df)
