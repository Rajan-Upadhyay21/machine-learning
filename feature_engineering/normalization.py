import pandas as pd
from sklearn.preprocessing import Normalizer

# Sample numerical data
data = {
    "Math": [80, 60, 90],
    "Science": [70, 65, 95],
    "English": [75, 70, 85]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Apply Normalization
normalizer = Normalizer()
normalized_data = normalizer.fit_transform(df)

normalized_df = pd.DataFrame(normalized_data, columns=df.columns)

print("\nNormalized Dataset:")
print(normalized_df)
