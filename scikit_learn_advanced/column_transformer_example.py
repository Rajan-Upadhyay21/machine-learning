# ---------------------------------------------------------
# Program: Column Transformer Example
# Description:
# This program demonstrates handling different column types
# using ColumnTransformer.
# It applies scaling to numeric data and one-hot encoding
# to categorical data in one workflow.
# ---------------------------------------------------------

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Sample mixed dataset
data = pd.DataFrame({
    "age": [23, 25, 21, 30],
    "salary": [50000, 60000, 45000, 80000],
    "department": ["AI", "ML", "AI", "DS"]
})

# Define column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["age", "salary"]),
        ("cat", OneHotEncoder(), ["department"])
    ]
)

# Fit and transform data
transformed_data = preprocessor.fit_transform(data)

print("Original data:")
print(data)

print("\nTransformed data shape:", transformed_data.shape)
print("Transformation completed successfully.")
