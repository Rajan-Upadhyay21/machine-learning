import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample categorical data
data = {
    "Education_Level": ["High School", "Bachelor", "Master", "PhD", "Bachelor", "Master"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Apply Label Encoding
label_encoder = LabelEncoder()
df["Education_Level_Encoded"] = label_encoder.fit_transform(df["Education_Level"])

print("\nDataset After Label Encoding:")
print(df)

print("\nLabel Mapping:")
for class_name, encoded_value in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
    print(f"{class_name} -> {encoded_value}")
