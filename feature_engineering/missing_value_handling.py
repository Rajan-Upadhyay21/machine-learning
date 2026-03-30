import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Sample dataset with missing values
data = {
    "Age": [25, 30, np.nan, 35, 40, np.nan],
    "Salary": [50000, 60000, 55000, np.nan, 65000, 70000],
    "Department": ["HR", "IT", "Finance", np.nan, "IT", "HR"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\nMissing Values Count:")
print(df.isnull().sum())

# Fill numerical missing values with mean
mean_imputer = SimpleImputer(strategy="mean")
df["Age"] = mean_imputer.fit_transform(df[["Age"]])
df["Salary"] = mean_imputer.fit_transform(df[["Salary"]])

# Fill categorical missing values with most frequent value
mode_imputer = SimpleImputer(strategy="most_frequent")
df["Department"] = mode_imputer.fit_transform(df[["Department"]]).ravel()

print("\nDataset After Handling Missing Values:")
print(df)

print("\nMissing Values Count After Imputation:")
print(df.isnull().sum())
