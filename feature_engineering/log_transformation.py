import pandas as pd
import numpy as np

# Sample skewed data
data = {
    "Income": [20000, 25000, 30000, 50000, 100000, 250000, 500000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Apply log transformation
df["Log_Income"] = np.log(df["Income"])

print("\nDataset After Log Transformation:")
print(df)
