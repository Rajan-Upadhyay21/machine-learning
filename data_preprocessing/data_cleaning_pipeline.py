import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = {
    "Age": [25, 30, np.nan, 35, 40, 30],
    "Salary": [50000, 60000, 55000, np.nan, 65000, 60000],
    "Department": ["HR", "IT", "Finance", np.nan, "IT", "IT"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

df = df.drop_duplicates()

mean_imputer = SimpleImputer(strategy="mean")
df["Age"] = mean_imputer.fit_transform(df[["Age"]])
df["Salary"] = mean_imputer.fit_transform(df[["Salary"]])

mode_imputer = SimpleImputer(strategy="most_frequent")
df["Department"] = mode_imputer.fit_transform(df[["Department"]]).ravel()

label_encoder = LabelEncoder()
df["Department"] = label_encoder.fit_transform(df["Department"])

scaler = StandardScaler()
df[["Age", "Salary"]] = scaler.fit_transform(df[["Age", "Salary"]])

print("\nCleaned Dataset:")
print(df)
