import pandas as pd

data = {
    "Department": ["HR", "IT", "Finance", "IT", "HR"],
    "City": ["Chicago", "Boston", "New York", "Boston", "Chicago"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

encoded_df = pd.get_dummies(df, columns=["Department", "City"])

print("\nEncoded Dataset:")
print(encoded_df)
