import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Alice", "David", "Bob"],
    "Age": [25, 30, 25, 40, 30],
    "City": ["Chicago", "New York", "Chicago", "Boston", "New York"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

df_no_duplicates = df.drop_duplicates()

print("\nDataset After Removing Duplicates:")
print(df_no_duplicates)
