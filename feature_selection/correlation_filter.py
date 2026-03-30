import pandas as pd
import numpy as np

data = {
    "Feature1": [1, 2, 3, 4, 5],
    "Feature2": [2, 4, 6, 8, 10],
    "Feature3": [5, 3, 6, 2, 1],
    "Feature4": [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

corr_matrix = df.corr().abs()
print("\nCorrelation Matrix:")
print(corr_matrix)

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)]

filtered_df = df.drop(columns=to_drop)

print("\nColumns Removed Due to High Correlation:")
print(to_drop)

print("\nDataset After Correlation Filtering:")
print(filtered_df)
